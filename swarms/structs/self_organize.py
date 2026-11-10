"""
SelfOrganizingSwarm: a leaderless, role-free, asynchronous swarm.

Inference-only port of s3, "Simple Scaling of Swarms via Cooperative
Multi-agent RL" (Jha, Faria, Li, Geng et al., 2026,
https://scalingswarms.com). N copies of one model share one objective
and coordinate only through asynchronous messages.
"""

from __future__ import annotations

import functools
import json
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Dict, List, Optional

from loguru import logger

from swarms.structs.agent import Agent
from swarms.prompts.self_organize_prompts import (
    CHECK_NOTIFICATIONS_TOOL_DOC,
    COLLABORATION_PROMPT,
    COST_LINE,
    DEFAULT_FINAL_PROMPT,
    FINAL_ANSWER_RULES,
    IDLE_RULES,
    MESSAGE_TOOL_DOC,
    MESSAGING_RULES,
    READ_MESSAGES_TOOL_DOC,
    SHARED_OBJECTIVE_PROMPT,
    SOLO_IDLE_RULES,
    SOLO_PROMPT,
)

_FINAL_RE = re.compile(r"^\s*####\s*(.+?)\s*$", re.MULTILINE)


def _final_answer(text: str) -> Optional[str]:
    """Return the last #### answer line in a reply, if any.

    Args:
        text: Model reply that may contain a final-answer marker.

    Returns:
        The answer string after the last ####, or None.
    """
    matches = _FINAL_RE.findall(text or "")
    return matches[-1] if matches else None


def _output_tokens(agent: Any) -> int:
    """Return generated tokens so far, as reported by the provider.

    Args:
        agent: Agent whose usage dict may contain output_tokens.

    Returns:
        Non-negative token count, or 0 when usage is unavailable.
    """
    try:
        return int(agent.usage.get("output_tokens", 0))
    except Exception:
        return 0


class MessageBoard:
    """Unread inboxes for one team episode.

    Supports direct messages to a named teammate or broadcasts with to="all".
    A shared threading.Condition wakes idle members when mail arrives.

    Args:
        names: Agent names that each get an inbox.
        cond: Condition shared with the episode (backed by an RLock).
        verbose: When True, log every delivered message.
    """

    def __init__(
        self,
        names: List[str],
        cond: threading.Condition,
        verbose: bool = False,
    ):
        self.names = list(names)
        self.cond = cond
        self.inbox: Dict[str, List[dict]] = {n: [] for n in names}
        self.pending: Dict[str, bool] = {n: False for n in names}
        self.log: List[dict] = []
        self.verbose = verbose
        self._t0 = time.time()

    def post(self, sender: str, to: str, text: str) -> str:
        """Deliver a message to one teammate or every other teammate.

        Args:
            sender: Name of the sending agent.
            to: Recipient name, or "all" for a broadcast.
            text: Message body.

        Returns:
            A short status string for the sender (delivery confirmation or error).
        """
        with self.cond:
            if to == "all":
                recipients = [n for n in self.names if n != sender]
            elif to in self.names and to != sender:
                recipients = [to]
            else:
                valid = ", ".join(
                    n for n in self.names if n != sender
                )
                return f"Unknown recipient {to!r}. Use one of: {valid}, or all."
            if not recipients:
                return "You have no teammates to message."

            msg = {
                "t": round(time.time() - self._t0, 2),
                "from": sender,
                "to": to,
                "text": text,
            }
            self.log.append(msg)
            for recipient in recipients:
                self.inbox[recipient].append(msg)
                self.pending[recipient] = True
            self.cond.notify_all()

        if self.verbose:
            logger.info(f"[{sender} -> {to}] {text}")
        return f"Delivered to {', '.join(recipients)}."

    def read(self, name: str, limit: int = 16) -> List[dict]:
        """Pop up to limit unread messages for name.

        Args:
            name: Agent whose inbox to drain.
            limit: Maximum messages to return in one call.

        Returns:
            The drained messages, oldest first. Clears pending when empty.
        """
        with self.cond:
            batch, self.inbox[name] = (
                self.inbox[name][:limit],
                self.inbox[name][limit:],
            )
            if not self.inbox[name]:
                self.pending[name] = False
            return batch

    def unread_counts(self, name: str) -> Dict[str, int]:
        """Count unread messages for name, grouped by sender.

        Args:
            name: Agent whose inbox to inspect.

        Returns:
            Mapping of sender name to unread count.
        """
        with self.cond:
            return dict(Counter(m["from"] for m in self.inbox[name]))

    def notification(self, name: str) -> Optional[str]:
        """Take the pending unread-count notice for name, if any.

        Matches the paper's notification format: counts only, never contents.

        Args:
            name: Agent to notify.

        Returns:
            JSON string with unread_by_sender and notice, or None.
        """
        with self.cond:
            if not self.pending[name]:
                return None
            self.pending[name] = False
            counts = self.unread_counts(name)

        if not counts:
            return None

        total = sum(counts.values())
        plural = "s" if total != 1 else ""
        notice = (
            f"You have {total} unread message{plural} from "
            f"{', '.join(counts)}.\nUse read_messages to read them; this "
            "notification does not mark messages as read."
        )
        return json.dumps(
            {"unread_by_sender": counts, "notice": notice}
        )


class _Episode:
    """Shared mutable state for one swarm attempt.

    Coordinates member status, the message board, submitted answers, and the
    work → final → done phase machine across threads.

    Args:
        names: Agent names participating in this episode.
        verifier: Optional predicate that returns True for a correct answer.
        stop_on_success: End early when a verified answer is submitted.
        timeout: Wall-clock seconds allowed for the work phase.
        verbose: Forwarded to the message board.
    """

    def __init__(
        self,
        names: List[str],
        verifier: Optional[Callable[[str], bool]],
        stop_on_success: bool,
        timeout: float,
        verbose: bool,
    ):
        self.cond = threading.Condition()
        self.board = MessageBoard(names, self.cond, verbose)
        self.status = {
            n: "working" for n in names
        }  # working | idle | finished
        self.answers: Dict[str, str] = {}
        self.solved_by: Optional[str] = None
        self.solution: Optional[str] = None
        self.env_solved = False
        self.phase = "work"  # work -> final -> done
        self.verifier = verifier
        self.stop_on_success = stop_on_success
        self.deadline = time.time() + timeout

    def submit(self, name: str, answer: str) -> None:
        """Record answer from name and optionally mark the team solved.

        Args:
            name: Submitting agent.
            answer: Proposed final answer.
        """
        ok = False
        if self.verifier is not None:
            try:
                ok = bool(self.verifier(answer))
            except Exception as e:
                logger.warning(
                    f"Verifier failed on {name}'s answer: {e}"
                )

        with self.cond:
            self.answers.setdefault(name, answer)
            if ok and self.solved_by is None:
                self.solved_by, self.solution = name, answer
                if self.stop_on_success:
                    self.phase = "done"
            self.cond.notify_all()

    def solve(self, name: str, answer: str = "") -> None:
        """Mark the episode solved via a task-environment tool.

        Args:
            name: Agent whose tool called solve.
            answer: Optional answer string from the environment.
        """
        with self.cond:
            if self.solved_by is None:
                self.solved_by, self.solution = name, answer
            self.env_solved = True
            self.phase = "done"
            self.cond.notify_all()

    def check_deadline(self) -> None:
        """Move from work to final when the wall-clock timeout has elapsed."""
        with self.cond:
            if self.phase == "work" and time.time() > self.deadline:
                self.phase = "final"
                self.cond.notify_all()

    def quiet(self) -> bool:
        """Return whether every member is idle/finished with no pending mail.

        Returns:
            True when the work phase can safely end for lack of activity.
        """
        with self.cond:
            no_workers = all(
                s != "working" for s in self.status.values()
            )
            idle_waiting = any(
                s == "idle" and self.board.pending[n]
                for n, s in self.status.items()
            )
            return no_workers and not idle_waiting


@dataclass
class _Member:
    """Per-agent bookkeeping for one episode.

    Attributes:
        name: Public agent name for this attempt.
        agent: The swarms Agent instance, set during build.
        tool_calls: Total tool invocations attributed to this member.
        calls_seen: Tool-call count at the end of the previous loop.
        out_of_budget: Set when the generation allowance is exhausted.
        lock: Guards tool_calls across concurrent tool wrappers.
    """

    name: str
    agent: Any = None
    tool_calls: int = 0
    calls_seen: int = 0
    out_of_budget: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


def _counted(fn: Callable, member: _Member) -> Callable:
    """Wrap a tool so the harness can detect whether a turn made a call.

    Errors are returned as text. If a tool raised, swarms would retry the whole
    batch of calls in that turn and deliver messages twice.

    Args:
        fn: Tool callable to wrap.
        member: Member whose tool_calls counter is incremented.

    Returns:
        A wrapper with the same name that increments the counter and catches
        exceptions.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with member.lock:
            member.tool_calls += 1
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return f"Error in {fn.__name__}: {e}"

    return wrapper


class SelfOrganizingSwarm:
    """A leaderless, role-free, asynchronous swarm with one shared objective.

    Args:
        name: Name of the swarm.
        description: What the swarm is for.
        num_agents: Team size N. Agents are rebuilt for every run.
        model_name: Model for every member (one policy, N copies).
        communication: False runs N independent agents (best-of-N baseline).
        max_tokens_per_agent: Per-agent generation allowance L. None for
            no limit.
        token_cost: λ in the team reward. 0 turns the cost term off.
        max_loops_per_wake: Safety cap on LLM calls between two idle periods;
            the token allowance and timeout are the real limits.
        timeout: Wall-clock limit for the work phase, in seconds.
        stop_on_success: End the episode as soon as an answer passes the
            verifier.
        seed: Seeds the agent names and teammate order.
        agent_kwargs: Extra keyword arguments passed to every swarms Agent.
        verbose: Log every message as it is sent.
    """

    def __init__(
        self,
        name: str = "SelfOrganizingSwarm",
        description: str = (
            "Leaderless swarm that coordinates through asynchronous messages."
        ),
        num_agents: int = 4,
        model_name: str = "gpt-5.4",
        communication: bool = True,
        max_tokens_per_agent: Optional[int] = 8192,
        token_cost: float = 0.25,
        max_loops_per_wake: int = 100,
        timeout: float = 900.0,
        stop_on_success: bool = True,
        seed: Optional[int] = None,
        agent_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ):
        if num_agents < 1:
            raise ValueError("num_agents must be at least 1")

        self.name = name
        self.description = description
        self.num_agents = num_agents
        self.model_name = model_name
        self.communication = communication
        self.max_tokens_per_agent = max_tokens_per_agent
        self.token_cost = token_cost
        self.max_loops_per_wake = max_loops_per_wake
        self.timeout = timeout
        self.stop_on_success = stop_on_success
        self.rng = random.Random(seed)
        self.agent_kwargs = agent_kwargs or {}
        self.verbose = verbose

    def run(
        self,
        task: str,
        private_inputs: Optional[List[str]] = None,
        verifier: Optional[Callable[[str], bool]] = None,
        task_tools: Optional[
            Callable[[str, Callable[..., None]], List[Callable]]
        ] = None,
        rules: str = "",
        final_prompt: Optional[str] = DEFAULT_FINAL_PROMPT,
    ) -> Dict[str, Any]:
        """Run one team attempt.

        Args:
            task: The problem, shown to every member as its first user turn.
            private_inputs: One string per member, appended to its first turn.
            verifier: Returns True for a correct final answer. Enables
                success, early stopping and the team reward.
            task_tools: Factory (agent_name, solve) -> tools for that
                member's private copy of the environment. Calling
                solve(answer) marks the team as solved and ends the episode.
            rules: Task-specific rules added to every system prompt.
            final_prompt: Asked of every member without an answer when the work
                phase ends. None skips the final-answer phase.

        Returns:
            Dict with the team answer, success, per-agent answers and tokens,
            the full message log and the team reward.
        """
        n = self.num_agents
        if private_inputs is not None and len(private_inputs) != n:
            raise ValueError(
                f"Expected {n} private inputs, got {len(private_inputs)}"
            )

        # Public names sampled per episode so no convention can hang on a name.
        names = self.rng.sample(
            [f"agent_{i}" for i in range(1, max(64, n) + 1)], n
        )
        ep = _Episode(
            names,
            verifier,
            self.stop_on_success,
            self.timeout,
            self.verbose,
        )
        members = [_Member(name) for name in names]
        for member in members:
            member.agent = self._build_agent(
                member, names, ep, task_tools, rules
            )

        first_turns = [
            (
                task
                if private_inputs is None
                else f"{task}\n\n{private_inputs[i]}"
            )
            for i in range(n)
        ]

        start = time.time()
        with ThreadPoolExecutor(max_workers=n) as pool:
            list(
                pool.map(
                    lambda pair: self._work(ep, *pair),
                    zip(members, first_turns),
                )
            )

        with ep.cond:
            if ep.phase == "work":
                ep.phase = "final"

        if ep.phase == "final" and final_prompt:
            waiting = [m for m in members if m.name not in ep.answers]
            if waiting:
                with ThreadPoolExecutor(
                    max_workers=len(waiting)
                ) as pool:
                    list(
                        pool.map(
                            lambda m: self._ask_final(
                                m, final_prompt
                            ),
                            waiting,
                        )
                    )

        ep.phase = "done"
        return self._report(
            ep, members, names, verifier, time.time() - start
        )

    def _work(
        self, ep: _Episode, member: _Member, first_turn: str
    ) -> None:
        """Run one member's work phase: act, idle, wake on mail, repeat.

        Args:
            ep: Shared episode state.
            member: This thread's member.
            first_turn: Initial user message for the first Agent.run.
        """
        turn = first_turn
        try:
            while True:
                member.agent.run(turn)
                with ep.cond:
                    if (
                        ep.phase != "work"
                        or member.name in ep.answers
                        or member.out_of_budget
                    ):
                        break

                    ep.status[member.name] = "idle"
                    ep.cond.notify_all()

                    while (
                        ep.phase == "work"
                        and not ep.board.pending[member.name]
                        and not ep.quiet()
                    ):
                        ep.cond.wait(timeout=0.5)
                        ep.check_deadline()

                    if (
                        ep.phase != "work"
                        or not ep.board.pending[member.name]
                    ):
                        break

                    ep.status[member.name] = "working"
                    turn = ep.board.notification(member.name)
        except Exception as e:
            logger.error(f"{member.name} stopped with an error: {e}")
        finally:
            with ep.cond:
                ep.status[member.name] = "finished"
                ep.cond.notify_all()

    def _ask_final(self, member: _Member, final_prompt: str) -> None:
        """Prompt a member that has not yet submitted an answer.

        Args:
            member: Member to ask.
            final_prompt: User message requesting a #### answer.
        """
        try:
            member.agent.max_loops = 1
            member.agent.run(final_prompt)
        except Exception as e:
            logger.error(
                f"{member.name} failed to give a final answer: {e}"
            )

    def _make_stop(
        self, ep: _Episode, member: _Member
    ) -> Callable[[Any], bool]:
        """Build the per-loop stopping callback for a member's Agent.

        Returning True ends the current Agent.run. Returning False
        lets it call the model again after any new-message notification has
        been added to its history.

        Args:
            ep: Shared episode state.
            member: Member this callback belongs to.

        Returns:
            A stopping_func suitable for Agent.
        """

        def should_stop(response: Any) -> bool:
            text = response if isinstance(response, str) else ""
            answer = _final_answer(text)

            # Lenient in the final phase: take the last non-empty line.
            if (
                ep.phase == "final"
                and answer is None
                and text.strip()
            ):
                answer = text.strip().splitlines()[-1]

            if answer is not None:
                ep.submit(member.name, answer)
                return True
            if ep.phase != "work":
                return True

            with member.lock:
                made_call = member.tool_calls > member.calls_seen
                member.calls_seen = member.tool_calls

            # Ended without a tool call → become idle.
            if not (
                made_call or (isinstance(response, list) and response)
            ):
                return True

            if (
                self.max_tokens_per_agent
                and _output_tokens(member.agent)
                >= self.max_tokens_per_agent
            ):
                member.out_of_budget = True
                return True

            ep.check_deadline()
            if ep.phase != "work":
                return True

            note = ep.board.notification(member.name)
            if note:
                member.agent.short_memory.add(
                    role="Harness", content=note
                )
            return False

        return should_stop

    def _messaging_tools(
        self, ep: _Episode, member: _Member
    ) -> List[Callable]:
        """Build message, read_messages and check_notifications.

        Args:
            ep: Shared episode state (owns the board).
            member: Member these tools are bound to.

        Returns:
            The three messaging callables for this member.
        """
        board = ep.board

        def message(
            to: Annotated[
                str, 'A teammate\'s name, or "all" for every teammate'
            ],
            text: Annotated[str, "The message to send"],
        ) -> str:
            return board.post(member.name, str(to).strip(), str(text))

        message.__doc__ = MESSAGE_TOOL_DOC

        def read_messages() -> str:
            msgs = board.read(member.name)
            if not msgs:
                return "No unread messages."
            return json.dumps(
                [
                    {
                        "from": item["from"],
                        "to": item["to"],
                        "text": item["text"],
                    }
                    for item in msgs
                ]
            )

        read_messages.__doc__ = READ_MESSAGES_TOOL_DOC

        def check_notifications() -> str:
            return json.dumps(
                {"unread_by_sender": board.unread_counts(member.name)}
            )

        check_notifications.__doc__ = CHECK_NOTIFICATIONS_TOOL_DOC
        return [message, read_messages, check_notifications]

    def _system_prompt(
        self, name: str, teammates: List[str], rules: str
    ) -> str:
        """Assemble the system prompt for one member.

        Args:
            name: This member's public name.
            teammates: Other members' names, already shuffled.
            rules: Optional task-specific rules block.

        Returns:
            The full system prompt string.
        """
        cost = (
            COST_LINE.format(lam=self.token_cost)
            if self.max_tokens_per_agent and self.token_cost
            else ""
        )

        if self.communication and teammates:
            identity = (
                f"You are {name}. Your teammates, in no particular order: "
                f"{', '.join(teammates)}."
            )
        else:
            identity = f"You are {name}."

        parts = [
            (
                COLLABORATION_PROMPT
                if self.communication
                else SOLO_PROMPT
            ),
            f"Task rules:\n{rules.strip()}" if rules.strip() else "",
            SHARED_OBJECTIVE_PROMPT.format(cost_line=cost),
            identity,
            MESSAGING_RULES if self.communication else "",
            (
                f"Your generation allowance is {self.max_tokens_per_agent} tokens."
                if self.max_tokens_per_agent
                else ""
            ),
            IDLE_RULES if self.communication else SOLO_IDLE_RULES,
            FINAL_ANSWER_RULES,
        ]
        return "\n\n".join(part for part in parts if part)

    def _build_agent(
        self,
        member: _Member,
        names: List[str],
        ep: _Episode,
        task_tools: Optional[
            Callable[[str, Callable[..., None]], List[Callable]]
        ],
        rules: str,
    ) -> Agent:
        """Construct the swarms Agent for one member of this episode.

        Args:
            member: Member whose agent is being built.
            names: All agent names in the episode.
            ep: Shared episode state.
            task_tools: Optional factory for environment tools.
            rules: Task-specific rules for the system prompt.

        Returns:
            A configured Agent ready to run.
        """
        teammates = [n for n in names if n != member.name]
        self.rng.shuffle(teammates)

        tools = (
            self._messaging_tools(ep, member)
            if self.communication
            else []
        )
        if task_tools is not None:
            tools += task_tools(
                member.name,
                lambda answer="", _n=member.name: ep.solve(
                    _n, answer
                ),
            )
        tools = [_counted(t, member) for t in tools]

        kwargs = {
            "print_on": False,
            # swarms' loop prompt would override the harness rules
            "reasoning_prompt_on": False,
            # each attempt starts from a clean history
            "persistent_memory": False,
            # a retry re-runs every call in the turn
            "tool_retry_attempts": 1,
        }
        kwargs.update(self.agent_kwargs)

        return Agent(
            agent_name=member.name,
            agent_description=f"Member of {self.name}",
            system_prompt=self._system_prompt(
                member.name, teammates, rules
            ),
            model_name=self.model_name,
            max_loops=self.max_loops_per_wake,
            tools=tools or None,
            stopping_func=self._make_stop(ep, member),
            **kwargs,
        )

    def _report(
        self,
        ep: _Episode,
        members: List[_Member],
        names: List[str],
        verifier: Optional[Callable[[str], bool]],
        elapsed: float,
    ) -> Dict[str, Any]:
        """Build the result dict after an episode finishes.

        Args:
            ep: Completed episode state.
            members: All members that participated.
            names: Agent names used this episode.
            verifier: Optional answer checker (affects success).
            elapsed: Wall-clock seconds for the attempt.

        Returns:
            Result mapping with answer, success, tokens, messages and reward.
        """
        tokens = {m.name: _output_tokens(m.agent) for m in members}
        n = len(members)

        if ep.solution is not None:
            answer = ep.solution
        elif ep.answers:
            answer = Counter(ep.answers.values()).most_common(1)[0][0]
        else:
            answer = None

        graded = verifier is not None or ep.env_solved
        success = (ep.solved_by is not None) if graded else None

        team_reward = None
        if success is not None:
            cost = 0.0
            if self.max_tokens_per_agent:
                cost = min(
                    1.0,
                    sum(tokens.values())
                    / (n * self.max_tokens_per_agent),
                )
            team_reward = float(success) * (
                1.0 - self.token_cost * cost
            )

        log = ep.board.log
        return {
            "answer": answer,
            "success": success,
            "solved_by": ep.solved_by,
            "answers": dict(ep.answers),
            "team_reward": team_reward,
            "tokens": tokens,
            "total_tokens": sum(tokens.values()),
            "message_calls": len(log),
            "broadcasts": sum(1 for x in log if x["to"] == "all"),
            "messages": list(log),
            "agents": names,
            "communication": self.communication,
            "elapsed_seconds": round(elapsed, 2),
        }
