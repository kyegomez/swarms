"""
Typed stream payloads for Swarms agents.

Content tokens travel through ``streaming_callback`` — and out of
``Agent.run_stream`` / ``Agent.arun_stream`` — as plain strings, and in
detailed streaming (``Agent(stream=True)``) as ``token_info`` dicts. Reasoning
("thinking") deltas from providers that emit ``delta.reasoning_content`` are
swallowed by default, exactly as they always were.

When ``Agent(stream_thinking=True)`` is set, those reasoning deltas are
forwarded down the same surface wrapped in :class:`ThinkingToken`. The wrapper
type is the distinguishing marker: it is neither ``str`` nor ``dict``, so a
consumer separates reasoning from output with a single ``isinstance`` check and
no string sniffing, and a consumer that never opts in keeps receiving exactly
what it received before.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ThinkingToken:
    """A reasoning delta streamed alongside an agent's content tokens."""

    content: str

    def __str__(self) -> str:
        return self.content
