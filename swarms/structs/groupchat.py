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

Flow (turn-based, one speaker per turn):

    task -> posted to the shared conversation; every agent sees it
    -> each turn, all agents privately "bid" via respond(score, message):
       a self-rated desire to speak plus the reply they would give
    -> the single highest bidder above `threshold` takes the floor; only its
       message is posted to the conversation
    -> a recency penalty discourages the same agent from speaking twice in a
       row, so the floor passes around the room
    -> stop when no agent bids above `threshold` for a turn (a conversational
       lull), or `max_loops` total messages have been posted

Only one agent speaks per turn, mirroring human turn-taking: everyone listens,
the most motivated/relevant participant jumps in, and the rest stay silent
unless they have something better to add.

Key concepts:

    RESPOND_TOOL
        A forced function-calling schema used to make every agent return a
        structured `(score, message)` bid instead of free-form text.

    threshold
        The minimum (recency-adjusted) score required to take the floor.
        Raising it makes the room more selective; lowering it livelier.

    recency_penalty / recency_window
        How much to subtract from the bid of an agent that spoke within the
        last `recency_window` turns. Prevents one agent from monologuing.

    max_loops
        A hard cap on total posted messages, including the initial user task.

Example:

    from swarms import Agent

    agents = [
        Agent(agent_name="Researcher", model_name="gpt-5.4"),
        Agent(agent_name="Critic", model_name="gpt-5.4"),
    ]

    chat = GroupChat(agents=agents, max_loops=10, threshold=0.6)
    result = chat.run("Discuss the tradeoffs of autonomous multi-agent systems.")
"""

import asyncio
import json
from collections import deque
from typing import Any, Callable, List, Optional, Tuple

from swarms.prompts.groupchat_prompt import GROUPCHAT_DECIDE_PROMPT
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.serialization import SerializableMixin
from swarms.utils.formatter import formatter
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="groupchat")

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

    # Providers return tool calls in two shapes: a plain dict
    # ``{"function": {"name", "arguments"}}`` (the MCP-normalized form), or a
    # raw provider object such as litellm's ``ChatCompletionMessageToolCall``
    # where ``function`` and ``arguments`` are *attributes*, not keys. Normalize
    # any pydantic-style object to a dict so both shapes parse identically.
    if not isinstance(tool_output, dict) and hasattr(
        tool_output, "model_dump"
    ):
        try:
            tool_output = tool_output.model_dump()
        except Exception:
            pass

    if isinstance(tool_output, dict):
        fn = tool_output.get("function")
    else:
        fn = getattr(tool_output, "function", None)
    if not fn:
        return 0.0, ""

    if isinstance(fn, dict):
        args = fn.get("arguments")
    else:
        args = getattr(fn, "arguments", None)
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
    """Coordinate a turn-based, self-selecting agent groupchat.

    Each turn every agent privately bids on whether to speak; the single
    highest (recency-adjusted) bidder above ``threshold`` takes the floor and
    its reply is the only message posted. This mirrors human turn-taking:
    everyone listens to the same conversation, the most motivated participant
    jumps in, and the rest stay silent unless they can do better next turn.

    There is no fixed speaking order — who speaks emerges from the bids — but a
    ``recency_penalty`` discourages one agent from monologuing so the floor
    moves around the room.

    The conversation stops when either:

    - ``max_loops`` total messages have been posted, or
    - no agent bids above ``threshold`` for a turn (a conversational lull).
    """

    _to_dict_exclude = ("agents", "conversation")

    def __init__(
        self,
        name: str = "dynamic-groupchat",
        description: str = "Agents take turns; one speaker per turn.",
        agents: Optional[List[Agent]] = None,
        max_loops: int = 20,
        threshold: float = 0.5,
        recency_penalty: float = 0.3,
        recency_window: int = 1,
        idle_timeout: float = 8.0,
        output_type: str = "str-all-except-first",
        verbose: bool = False,
        auto_equip: bool = True,
    ):
        """Initialize the turn-based groupchat runtime.

        Args:
            name: Human-readable name used in logs and serialized state.
            description: Short description of the chat structure.
            agents: Agents participating in the conversation. At least two are
                required for a meaningful discussion.
            max_loops: Maximum number of messages posted before stopping. The
                initial user task counts as the first message.
            threshold: Minimum (recency-adjusted) bid required to take the
                floor. A turn where no agent clears it ends the chat.
            recency_penalty: Amount subtracted from the bid of any agent that
                spoke within the last ``recency_window`` turns. Discourages a
                single agent from monologuing. Set to ``0.0`` to disable.
            recency_window: How many of the most recent speakers are subject to
                ``recency_penalty``.
            idle_timeout: Deprecated/unused — the chat now ends on a bidding
                lull rather than a wall-clock timeout. Kept for compatibility.
            output_type: Format passed to ``history_output_formatter``.
            verbose: Whether to emit internal log messages and print each
                posted message as a panel to stdout.
            auto_equip: When ``True`` (default), automatically inject
                ``RESPOND_TOOL`` into any agent that doesn't already carry it,
                so every agent can produce a machine-readable speaking bid.
                Set to ``False`` if your agents already declare the tool
                themselves.

        Raises:
            ValueError: If fewer than two agents are provided.
        """
        self.name = name
        self.description = description
        self.agents = agents
        self.max_loops = max_loops
        self.threshold = threshold
        self.recency_penalty = recency_penalty
        self.recency_window = recency_window
        self.idle_timeout = idle_timeout
        self.output_type = output_type
        self.verbose = verbose
        self.auto_equip = auto_equip

        self.conversation = Conversation(time_enabled=True)

        if len(self.agents) < 2:
            raise ValueError("GroupChat requires at least 2 agents.")

        if self.auto_equip:
            self._ensure_respond_tool()

    def _ensure_respond_tool(self) -> None:
        """Inject ``RESPOND_TOOL`` into agents that do not already carry it.

        Without the ``respond`` tool an agent's speaking decision can never be
        parsed, so it would sit silent for the whole chat. The agent's LLM
        client bakes ``tools_list_dictionary`` in at construction time, so
        after appending the schema the client is rebuilt via
        ``llm_handling()``.
        """
        for agent in self.agents:
            tools = agent.tools_list_dictionary or []
            if any(
                tool.get("function", {}).get("name") == "respond"
                for tool in tools
                if isinstance(tool, dict)
            ):
                continue
            agent.tools_list_dictionary = [*tools, RESPOND_TOOL]
            agent.llm = agent.llm_handling()
            self._log(
                "info",
                f"Injected respond tool into {agent.agent_name}",
            )

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
        prompt = GROUPCHAT_DECIDE_PROMPT.format(
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
            # Surface failures unconditionally — a swallowed error here looks
            # exactly like "the agent chose to stay silent", which makes a bad
            # model name or missing API key impossible to diagnose.
            logger.warning(
                f"[{self.name}] {agent.agent_name} failed to bid: "
                f"{type(e).__name__}: {e}"
            )
            return 0.0, ""
        return _extract_args(tool_output)

    def _post(
        self,
        sender: str,
        content: str,
        score: Optional[float],
        streaming_callback: Optional[
            Callable[[str, str, bool], None]
        ] = None,
    ) -> None:
        """Record a message in the shared conversation and optionally print it.

        There are no per-agent inboxes: every agent reads the same
        ``Conversation`` when it builds its next bid, so a single append makes
        the message visible to everyone the following turn.

        When ``streaming_callback`` is provided, the posted message is replayed
        token-by-token to the callback before being recorded, so consumers can
        render the speaker's reply as it "arrives" — mirroring the live
        streaming of ``SequentialWorkflow`` / ``AgentRearrange`` where one agent
        speaks at a time.
        """
        if streaming_callback is not None:
            self._stream_reply(sender, content, streaming_callback)

        metadata = {"score": score} if score is not None else None
        self.conversation.add(
            role=sender, content=content, metadata=metadata
        )
        self._log(
            "info",
            f"{sender} -> {content[:80]} "
            f"(score={'-' if score is None else f'{score:.2f}'})",
        )
        if self.verbose:
            title = (
                f"{sender}"
                if score is None
                else f"{sender}  (score={score:.2f})"
            )
            style = "bold green" if score is None else "bold blue"
            formatter.print_panel(content, title=title, style=style)

    def _stream_reply(
        self,
        sender: str,
        content: str,
        streaming_callback: Callable[[str, str, bool], None],
    ) -> None:
        """Replay a posted message to ``streaming_callback`` as token chunks.

        The groupchat is turn-based — exactly one agent holds the floor per
        turn — so its reply is generated atomically (inside the bid) rather than
        streamed live. To still offer the token-over-time experience of the
        other swarm structures, the finished reply is chunked on whitespace and
        emitted one piece at a time, followed by a final empty-chunk sentinel.

        The callback signature matches the rest of the framework:
        ``streaming_callback(agent_name: str, chunk: str, is_final: bool)``.
        ``is_final=True`` marks the end of this speaker's turn.
        """
        words = content.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == len(words) - 1 else f"{word} "
            if chunk:
                streaming_callback(sender, chunk, False)
        streaming_callback(sender, "", True)

    async def _collect_bids(
        self, sender: str, message: str, history: str
    ) -> List[Tuple[Agent, float, str]]:
        """Ask every agent, concurrently, for a speaking bid this turn.

        Each agent returns a ``(score, message)`` pair: how much it wants the
        floor and the reply it would give. Bidding is the cheap "do I have
        something to add right now?" instinct — it runs in parallel for speed,
        but at most one bid is ever posted (see ``_select_speaker``). Agent
        calls are blocking, so each runs in its own thread via
        ``asyncio.to_thread`` and one slow model cannot stall the turn.
        """
        results = await asyncio.gather(
            *(
                asyncio.to_thread(
                    self._decide_sync, agent, sender, message, history
                )
                for agent in self.agents
            )
        )
        return [
            (agent, score, reply)
            for agent, (score, reply) in zip(self.agents, results)
        ]

    def _select_speaker(
        self,
        bids: List[Tuple[Agent, float, str]],
        recent: set,
    ) -> Optional[Tuple[Agent, float, str]]:
        """Pick the single agent that takes the floor this turn.

        The winner is the highest *recency-adjusted* bid that (a) carries a
        non-empty reply and (b) clears ``threshold``. Agents that spoke within
        the last ``recency_window`` turns have ``recency_penalty`` subtracted
        from their score, so the floor passes around instead of one agent
        monologuing. Returns ``None`` when nobody clears the bar — a lull that
        ends the conversation.

        Returns:
            ``(agent, raw_score, reply)`` for the chosen speaker, or ``None``.
        """
        best: Optional[Tuple[Agent, float, str]] = None
        best_adjusted = self.threshold

        for agent, score, reply in bids:
            if not reply:
                continue
            adjusted = score
            if agent.agent_name in recent:
                adjusted -= self.recency_penalty
            if adjusted <= best_adjusted:
                continue
            best_adjusted = adjusted
            best = (agent, score, reply)

        return best

    async def _run_async(
        self,
        task: str,
        streaming_callback: Optional[
            Callable[[str, str, bool], None]
        ] = None,
    ) -> Any:
        """Run the turn-based groupchat and return formatted history.

        Each turn: snapshot the shared history, collect a bid from every agent,
        let the single highest (recency-adjusted) bidder speak, and post only
        that reply. The loop stops at the first lull — a turn where no agent
        clears ``threshold`` — or once ``max_loops`` messages have been posted.

        Args:
            task: Initial user message that seeds the conversation.
            streaming_callback: Optional ``(agent_name, chunk, is_final)``
                callback. Each posted message — the initial user task and every
                speaker's reply — is streamed to it token-by-token, with an
                ``is_final=True`` sentinel marking the end of each turn.

        Returns:
            Conversation history formatted according to ``self.output_type``.
        """
        self._log("info", f"[{self.name}] initial task: {task}")

        self._post(
            sender="User",
            content=task,
            score=None,
            streaming_callback=streaming_callback,
        )
        last_sender, last_message = "User", task

        recent: "deque[str]" = deque(
            maxlen=max(1, self.recency_window)
        )
        message_count = 1  # the user task counts as the first message

        while message_count < self.max_loops:
            history = self.conversation.return_history_as_string()
            bids = await self._collect_bids(
                last_sender, last_message, history
            )

            selection = self._select_speaker(bids, set(recent))
            if selection is None:
                # Distinguish a genuine conversational lull from a misconfigured
                # room. If not a single agent produced a non-empty reply on the
                # very first turn, the cause is almost never "nobody had
                # anything to say" — it's a bad model name, missing API key, or
                # a model that can't make the forced ``respond`` tool call.
                if message_count == 1 and not any(
                    reply for _, _, reply in bids
                ):
                    logger.warning(
                        f"[{self.name}] No agent produced a reply on the first "
                        "turn. The chat will end immediately. Likely causes: an "
                        "invalid model_name, a missing/invalid API key, or a "
                        "model without function-calling support. Run with "
                        "verbose=True to see each agent's bid."
                    )
                else:
                    self._log(
                        "info",
                        "no agent cleared the threshold — lull, stopping",
                    )
                break

            agent, score, reply = selection
            self._post(
                sender=agent.agent_name,
                content=reply,
                score=score,
                streaming_callback=streaming_callback,
            )
            recent.append(agent.agent_name)
            last_sender, last_message = agent.agent_name, reply
            message_count += 1

        return history_output_formatter(
            conversation=self.conversation, type=self.output_type
        )

    def run(
        self,
        task: str,
        streaming_callback: Optional[
            Callable[[str, str, bool], None]
        ] = None,
    ) -> Any:
        """Synchronously run the groupchat until a lull or ``max_loops``.

        Args:
            task: Initial user task or message for the group.
            streaming_callback: Optional ``(agent_name, chunk, is_final)``
                callback that receives each posted message as a stream of token
                chunks, with ``is_final=True`` marking the end of a speaker's
                turn. Matches the streaming signature used across the framework
                (``ConcurrentWorkflow``, ``HierarchicalSwarm``, etc.).

        Returns:
            Formatted conversation output from ``_run_async``.
        """
        return asyncio.run(
            self._run_async(
                task, streaming_callback=streaming_callback
            )
        )

    def run_batch(self, tasks: List[str]) -> List[Any]:
        """Run the groupchat in batch mode.

        Args:
            tasks: List of user tasks or messages for the group.

        Returns:
            List of formatted conversation outputs from ``_run_async``.
        """
        return [self.run(task) for task in tasks]
