"""
Structured conversation transcript for agent loops.

Chat-completions APIs expect a conversation to be a sequence of *typed* turns:
a user message, an assistant message that may carry ``tool_calls``, and one
``{"role": "tool", "tool_call_id": ...}`` message per call the assistant made.
Models are post-trained on that shape, and the API enforces part of it - an
assistant message with ``tool_calls`` that is not answered by a tool message
for every id causes the next request to be rejected.

Flattening the conversation into a single user string loses all of it: the
model can no longer tell its own prior actions from the user's input, tool
results arrive as prose, and the stable prefix needed for prompt caching is
rebuilt on every turn.

This module owns that structure so the agent loops do not each reimplement it.
Given a ``conversation``, a ``Transcript`` also records the assistant's turns
and the tool results into it as typed rows, so ``short_memory`` carries the
same tool calls the model saw and persistence, rendering and other agents
reading it (:func:`~swarms.structs.context_utils.messages_for`) see them.
User turns are still recorded by the callers, which own their role names.
"""

from typing import Any, Dict, List, Optional


class Transcript:
    """
    An ordered list of chat-completions messages, built incrementally.

    The class exists mainly to guarantee one invariant: every tool call
    recorded by :meth:`record_assistant` receives a matching tool result before
    the next request goes out. :meth:`flush_tool_results` enforces it, filling
    in a placeholder for any call that was never dispatched so the model learns
    why rather than the request failing.

    Example:
        >>> t = Transcript()
        >>> t.append_user("read config.json")
        >>> calls = t.record_assistant(parsed_response)
        >>> results = {calls[0]["id"]: "{...}"} if calls else {}
        >>> t.flush_tool_results(calls, results)
        >>> messages = t.messages
    """

    def __init__(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        conversation: Any = None,
        agent_name: Optional[str] = None,
    ):
        """
        Args:
            messages: Turns to start from.
            conversation: A :class:`~swarms.structs.conversation.Conversation`
                that assistant turns and tool results are also written to.
            agent_name: The role those assistant turns carry there.
        """
        self._messages: List[Dict[str, Any]] = list(messages or [])
        self._conversation = conversation
        self._agent_name = agent_name

    # Reading.

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """A copy of the conversation, safe to hand to a request."""
        return list(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self):
        return iter(self._messages)

    def __getitem__(self, index):
        return self._messages[index]

    def __bool__(self) -> bool:
        return bool(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    # Writing.

    def append_message(self, message: Dict[str, Any]) -> None:
        """Add an already-typed turn, e.g. one read back from a Conversation."""
        self._messages.append(dict(message))

    def insert_message(
        self, index: int, message: Dict[str, Any]
    ) -> None:
        """Insert a typed turn at ``index``; tool results must directly follow their call."""
        self._messages.insert(index, dict(message))

    def append_user(self, content: Any) -> None:
        """Add a user turn."""
        self._messages.append(
            {"role": "user", "content": str(content)}
        )

    def append_assistant_text(self, content: Any) -> None:
        """Add a plain assistant turn carrying no tool calls."""
        self._messages.append(
            {"role": "assistant", "content": str(content)}
        )

    def record_assistant(self, parsed: Any) -> List[Dict[str, Any]]:
        """
        Add the model's turn and return the tool calls it made.

        Args:
            parsed: The model's response after ``Agent.parse_llm_output`` -
                either a list of tool-call dicts or plain text.

        Returns:
            The tool calls, normalised to ``{"id", "name", "arguments"}``.
            Empty for a text-only turn. Every id returned must be passed to
            :meth:`flush_tool_results` before the next request.
        """
        calls: List[Dict[str, Any]] = []

        if isinstance(parsed, list):
            tool_calls = []
            for index, item in enumerate(parsed):
                if not isinstance(item, dict):
                    continue
                function = item.get("function") or {}
                name = function.get("name")
                if not name:
                    continue
                # Providers normally supply an id; synthesise a stable one if
                # not, since result pairing depends on it.
                call_id = (
                    item.get("id")
                    or f"call_{len(self._messages)}_{index}"
                )
                arguments = function.get("arguments", "{}")
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments,
                        },
                    }
                )
                calls.append(
                    {
                        "id": call_id,
                        "name": name,
                        "arguments": arguments,
                    }
                )

            if tool_calls:
                self._messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    }
                )
                if self._conversation is not None:
                    self._conversation.add(
                        role=self._agent_name,
                        content=None,
                        tool_calls=tool_calls,
                    )
                return calls

        self.append_assistant_text(parsed)
        if self._conversation is not None:
            self._conversation.add(
                role=self._agent_name, content=parsed
            )
        return calls

    def flush_tool_results(
        self,
        calls: List[Dict[str, Any]],
        results: Dict[str, Any],
    ) -> None:
        """
        Emit exactly one tool result per call in the preceding assistant turn.

        Args:
            calls: What :meth:`record_assistant` returned.
            results: Results keyed by tool call id. Missing entries are filled
                with a placeholder rather than skipped, because a gap makes the
                next request invalid.
        """
        for call in calls:
            recorded = call["id"] in results
            result = results.get(
                call["id"],
                f"(no result recorded for {call['name']})",
            )
            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": str(result),
                }
            )
            # The placeholder only keeps the request valid, nothing happened
            if self._conversation is not None and recorded:
                self._conversation.add(
                    role="tool",
                    content=str(result),
                    tool_call_id=call["id"],
                    metadata={"name": call["name"]},
                )

    def map_batch_results(
        self,
        tool_calls: List[Dict[str, Any]],
        output: Any,
        results: Dict[str, Any],
        formatter=str,
    ) -> None:
        """
        Attribute a batched tool execution back to individual call ids.

        Batch executors return one value per call when they can and a single
        combined value otherwise. Either way every id needs an entry, so a
        non-list output is recorded against all of them rather than leaving
        gaps.
        """
        if isinstance(output, list) and len(output) == len(
            tool_calls
        ):
            pairs = zip(tool_calls, output)
        else:
            pairs = ((call, output) for call in tool_calls)

        for call, value in pairs:
            results[call.get("id", "")] = formatter(value)
