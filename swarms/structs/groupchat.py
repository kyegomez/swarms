"""
GroupChat
================

An asynchronous groupchat where each agent independently decides whether to
respond to messages, on its own schedule. There are no rounds — any agent (or
several agents at once) can chime in whenever they have something to say.

This module is useful when a discussion should feel more like an open room than
a turn-based panel. Each participant receives every message, evaluates whether
it has something useful to add, and only broadcasts a reply when its self-rated
desire to speak clears the configured threshold.

Flow:

    task -> initial message posted to every agent's inbox
    -> each agent's coroutine pulls messages, calls respond(score, message) via
       a forced function call, and broadcasts when score > threshold
    -> broadcasts wake every other agent's inbox concurrently
    -> stop when no messages have been produced for `idle_timeout` seconds,
       or `max_loops` total messages have been posted

Key concepts:

    RESPOND_TOOL
        A forced function-calling schema used to make every agent return a
        structured `(score, message)` decision instead of free-form text.

    threshold
        The minimum score required for a reply to be published. Raising this
        value makes agents more selective; lowering it creates livelier chats.

    idle_timeout
        The amount of quiet time, in seconds, after which the conversation ends.

    max_loops
        A hard cap on total posted messages, including the initial user task.

Example:

    from swarms import Agent

    agents = [
        Agent(agent_name="Researcher", model_name="gpt-4.1"),
        Agent(agent_name="Critic", model_name="gpt-4.1"),
    ]

    chat = GroupChat(agents=agents, max_loops=10, threshold=0.6)
    result = chat.run("Discuss the tradeoffs of autonomous multi-agent systems.")
"""

import asyncio
import json
import time
from typing import Any, List, Optional, Tuple

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.serialization import SerializableMixin
from swarms.utils.formatter import formatter
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)

RESPOND_TOOL = {
    # The LLM is forced to call this function for every decision. This keeps
    # speaking decisions machine-readable and avoids parsing natural language.
    "type": "function",
    "function": {
        "name": "respond",
        "description": (
            "Decide whether to reply in the groupchat. Set score 0..1 for how much "
            "you want to speak. If you don't want to speak, set message to empty string."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "description": "How much you want to respond (0 = silent, 1 = strongly want to).",
                    "minimum": 0,
                    "maximum": 1,
                },
                "message": {
                    "type": "string",
                    "description": "Your reply to the group, or empty string if you don't want to speak.",
                },
            },
            "required": ["score", "message"],
        },
    },
}

DECIDE_PROMPT = """You are {agent_name} in a groupchat with: {other_agents}.

Conversation so far:
{history}

Latest message from {sender}:
{message}

Decide whether to speak. Silence is the default — most messages do NOT warrant
a reply from you. Only respond when you genuinely add value.

Score high (>= 0.7) ONLY if:
  - The message is directly in your area of expertise AND you have something
    substantive to contribute that nobody else has said.
  - You're directly addressed or @-mentioned.
  - There's a factual error or weak claim you can sharpen or correct.
  - You can move the conversation forward with a concrete next step or question.

Score low (< 0.5) — stay silent — if:
  - The topic is outside your expertise.
  - Your point would echo or paraphrase something already said.
  - You'd only be adding agreement, encouragement, or filler ("great point",
    "I agree", "well said").
  - The conversation is already converging and you'd just pile on.
  - You spoke very recently and have nothing new to add.

Call the `respond` function. If score < 0.5, return an empty message.
Otherwise, give a tight, specific reply — no preamble, no restating others.
"""


def _extract_args(tool_output: Any) -> Tuple[float, str]:
    """Parse and normalize a forced ``respond()`` tool call.

    Args:
        tool_output: Raw provider output from the forced tool invocation. Some
            providers return a single tool-call dictionary, while others return
            a list of tool-call dictionaries.

    Returns:
        A ``(score, message)`` pair. Invalid or missing output is treated as a
        silent decision: ``(0.0, "")``. Scores are clamped into the ``0..1``
        range and messages are stripped of surrounding whitespace.
    """
    if isinstance(tool_output, list):
        tool_output = tool_output[0] if tool_output else None
    if not tool_output:
        return 0.0, ""

    fn = (
        tool_output.get("function")
        if isinstance(tool_output, dict)
        else None
    )
    if not fn:
        return 0.0, ""

    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return 0.0, ""
    if not isinstance(args, dict):
        return 0.0, ""

    try:
        score = float(args.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    message = str(args.get("message", "")).strip()
    return max(0.0, min(1.0, score)), message


class GroupChat(SerializableMixin):
    """Coordinate an asynchronous, self-selecting agent groupchat.

    ``GroupChat`` starts one coroutine per agent. Every coroutine listens
    to its own inbox, asks the corresponding agent model whether it should
    reply, and broadcasts qualifying replies to every other inbox.

    Unlike round-robin group chats, there is no global speaking order. Multiple
    agents can react to the same message at nearly the same time, and agents can
    remain silent when they have nothing useful to add.

    The conversation stops when either:

    - ``max_loops`` total messages have been posted.
    - No new messages arrive for ``idle_timeout`` seconds.
    """

    _to_dict_exclude = ("agents", "conversation")

    def __init__(
        self,
        name: str = "dynamic-groupchat",
        description: str = "Agents choose whether to speak at any time.",
        agents: Optional[List[Agent]] = None,
        max_loops: int = 20,
        threshold: float = 0.5,
        idle_timeout: float = 8.0,
        output_type: str = "str-all-except-first",
        verbose: bool = False,
        print_on: bool = True,
    ):
        """Initialize the dynamic groupchat runtime.

        Args:
            name: Human-readable name used in logs and serialized state.
            description: Short description of the chat structure.
            agents: Agents participating in the conversation. At least two are
                required because each message is broadcast to "other" agents.
            max_loops: Maximum number of messages posted before stopping. The
                initial user task counts as the first message.
            threshold: Minimum decision score required to publish a reply.
            idle_timeout: Seconds of inactivity before the chat stops.
            output_type: Format passed to ``history_output_formatter``.
            verbose: Whether to emit internal log messages.

        Raises:
            ValueError: If fewer than two agents are provided.
        """
        self.name = name
        self.description = description
        self.agents = agents or []
        self.max_loops = max_loops
        self.threshold = threshold
        self.idle_timeout = idle_timeout
        self.output_type = output_type
        self.verbose = verbose
        self.print_on = print_on

        self.conversation = Conversation(time_enabled=True)

        if len(self.agents) < 2:
            raise ValueError("GroupChat requires at least 2 agents.")

    def _other_agents(self, exclude: str) -> str:
        """Return a comma-separated list of peers visible to one agent."""
        return ", ".join(
            a.agent_name
            for a in self.agents
            if a.agent_name != exclude
        )

    def _decide_sync(
        self, agent: Agent, sender: str, message: str, history: str
    ) -> Tuple[float, str]:
        """Ask one agent whether it wants to respond to a message.

        Runs the agent (which carries the ``respond`` tool in its
        ``tools_list_dictionary``) and extracts the structured ``(score,
        message)`` from the resulting tool call. Synchronous because
        ``Agent.run`` is blocking; the async agent loop invokes this method
        via ``asyncio.to_thread`` so one slow model call cannot block the
        whole groupchat.
        """
        prompt = DECIDE_PROMPT.format(
            agent_name=agent.agent_name,
            other_agents=self._other_agents(agent.agent_name),
            history=history,
            sender=sender,
            message=message,
        )
        try:
            tool_output = agent.run(task=prompt)
            # print(f"Agent {agent.agent_name} response: {tool_output}")
        except Exception as e:
            self._log("warning", f"{agent.agent_name} failed: {e}")
            return 0.0, ""
        return _extract_args(tool_output)

    async def _broadcast(
        self,
        sender: str,
        content: str,
        score: Optional[float],
        inboxes: dict,
        state: dict,
    ) -> None:
        """Record a message and enqueue it for every other agent.

        The conversation write and shared counters are protected by
        ``state["lock"]``. Queue fan-out happens after the lock is released so
        waiting agents can process messages concurrently.
        """
        async with state["lock"]:
            metadata = {"score": score} if score is not None else None
            self.conversation.add(
                role=sender, content=content, metadata=metadata
            )
            state["message_count"] += 1
            state["last_activity"] = time.monotonic()
            self._log(
                "info",
                f"{sender} -> {content[:80]} "
                f"(score={score if score is None else f'{score:.2f}'}, "
                f"count={state['message_count']})",
            )
            if state["message_count"] >= self.max_loops:
                state["stop"].set()

        if self.print_on:
            title = (
                f"{sender}"
                if score is None
                else f"{sender}  (score={score:.2f})"
            )
            style = "bold green" if score is None else "bold blue"
            formatter.print_panel(content, title=title, style=style)

        for name, inbox in inboxes.items():
            if name == sender:
                continue
            await inbox.put((sender, content))

    async def _agent_loop(
        self,
        agent: Agent,
        inbox: "asyncio.Queue[Tuple[str, str]]",
        inboxes: dict,
        state: dict,
    ) -> None:
        """Run one agent's inbox-processing loop.

        Each agent waits for messages from other participants, snapshots the
        current conversation history, asks the model for a structured speaking
        decision, and broadcasts the reply if it clears ``threshold``.
        """
        stop: asyncio.Event = state["stop"]

        while not stop.is_set():
            try:
                sender, message = await asyncio.wait_for(
                    inbox.get(), timeout=0.5
                )
            except asyncio.TimeoutError:
                continue

            if stop.is_set():
                return

            async with state["lock"]:
                history = self.conversation.return_history_as_string()

            score, reply = await asyncio.to_thread(
                self._decide_sync, agent, sender, message, history
            )

            if score > self.threshold and reply:
                await self._broadcast(
                    sender=agent.agent_name,
                    content=reply,
                    score=score,
                    inboxes=inboxes,
                    state=state,
                )

    async def _idle_monitor(self, state: dict) -> None:
        """Set the stop event after ``idle_timeout`` seconds of inactivity."""
        stop: asyncio.Event = state["stop"]
        while not stop.is_set():
            await asyncio.sleep(0.5)
            if (
                time.monotonic() - state["last_activity"]
                > self.idle_timeout
            ):
                self._log(
                    "info",
                    f"idle for {self.idle_timeout}s — stopping",
                )
                stop.set()
                return

    async def _run_async(self, task: str) -> Any:
        """Run the groupchat asynchronously and return formatted history.

        Args:
            task: Initial user message that seeds every agent's inbox.

        Returns:
            Conversation history formatted according to ``self.output_type``.
        """
        self._log("info", f"[{self.name}] initial task: {task}")

        inboxes = {a.agent_name: asyncio.Queue() for a in self.agents}
        state = {
            "lock": asyncio.Lock(),
            "stop": asyncio.Event(),
            "last_activity": time.monotonic(),
            "message_count": 0,
        }

        await self._broadcast(
            sender="User",
            content=task,
            score=None,
            inboxes=inboxes,
            state=state,
        )

        agent_tasks = [
            asyncio.create_task(
                self._agent_loop(
                    a, inboxes[a.agent_name], inboxes, state
                )
            )
            for a in self.agents
        ]
        monitor_task = asyncio.create_task(self._idle_monitor(state))

        await state["stop"].wait()

        for t in agent_tasks:
            t.cancel()
        monitor_task.cancel()
        await asyncio.gather(
            *agent_tasks, monitor_task, return_exceptions=True
        )

        return history_output_formatter(
            conversation=self.conversation, type=self.output_type
        )

    def run(self, task: str) -> Any:
        """Synchronously run the groupchat until it becomes idle or capped.

        Args:
            task: Initial user task or message for the group.

        Returns:
            Formatted conversation output from ``_run_async``.
        """
        return asyncio.run(self._run_async(task))

    def run_batch(self, tasks: List[str]) -> List[Any]:
        """Run the groupchat in batch mode.

        Args:
            tasks: List of user tasks or messages for the group.

        Returns:
            List of formatted conversation outputs from ``_run_async``.
        """
        return [self.run(task) for task in tasks]
