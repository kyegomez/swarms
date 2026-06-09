"""Tests for real-time thinking token streaming via callbacks and arun_stream."""

from types import SimpleNamespace
from typing import Dict, List

import pytest

from swarms import Agent


def _delta(**kw):
    return SimpleNamespace(
        content=kw.get("content", None),
        reasoning_content=kw.get("reasoning_content", None),
        tool_calls=kw.get("tool_calls", None),
    )


def _chunk(delta):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=None)]
    )


def content_chunk(text: str):
    return _chunk(_delta(content=text))


def thinking_chunk(text: str):
    return _chunk(_delta(reasoning_content=text))


class TestStreamEventHelpers:
    def test_callback_accepts_stream_events_dict_annotation(self):
        def on_event(event: dict) -> None:
            pass

        assert Agent._callback_accepts_stream_events(on_event)

    def test_callback_accepts_stream_events_str_annotation(self):
        def on_event(event: Dict[str, str]) -> None:
            pass

        assert Agent._callback_accepts_stream_events(on_event)

    def test_str_callback_not_events(self):
        def on_token(token: str) -> None:
            pass

        assert not Agent._callback_accepts_stream_events(on_token)

    def test_use_stream_events_flag(self):
        agent = Agent(
            agent_name="Evt",
            model_name="gpt-5.4",
            streaming_events=True,
            print_on=False,
        )
        assert agent._use_stream_events(lambda t: None)


class TestExtractThinkingStreamEvents:
    def _agent(self, print_on: bool = False):
        return Agent(
            agent_name="ThinkBot",
            model_name="gpt-5.4",
            max_loops=1,
            print_on=print_on,
        )

    def test_thinking_events_emitted_in_real_time(self):
        agent = self._agent()
        events: List[dict] = []

        def sink(event: dict) -> None:
            events.append(event)

        chunks = [
            thinking_chunk("step "),
            thinking_chunk("two"),
            content_chunk("answer"),
        ]
        stream = agent._extract_thinking_from_stream(
            iter(chunks),
            stream_sink=sink,
            use_stream_events=True,
        )
        list(stream)

        types = [e["type"] for e in events]
        assert types.index("thinking_start") < types.index("thinking")
        assert types.count("thinking") == 2
        assert "thinking_end" in types
        assert types.index("thinking_end") == len(types) - 1
        assert events[-1]["text"] == "step two"

    def test_str_callback_not_called_for_thinking(self):
        agent = self._agent()
        tokens: List[str] = []

        stream = agent._extract_thinking_from_stream(
            iter([thinking_chunk("hidden"), content_chunk("ok")]),
            stream_sink=tokens.append,
            use_stream_events=False,
        )
        list(stream)
        # Thinking must not reach a str-only sink; content is yielded, not sunk here.
        assert tokens == []

    def test_content_chunks_still_yielded(self):
        agent = self._agent()
        out = list(
            agent._extract_thinking_from_stream(
                iter([thinking_chunk("t"), content_chunk("result")]),
                use_stream_events=True,
            )
        )
        assert any(
            hasattr(c, "choices")
            and c.choices[0].delta.content == "result"
            for c in out
        )


class TestCallLlmStreamingCallback:
    def test_call_llm_streams_thinking_then_content_events(self):
        from swarms.utils.formatter import formatter

        agent = Agent(
            agent_name="Reasoner",
            model_name="gpt-5.4",
            streaming_on=True,
            print_on=False,
            streaming_events=True,
        )
        events: List[dict] = []

        class FakeLLM:
            stream = True

            def run(self, task, **kwargs):
                return iter(
                    [
                        thinking_chunk("reason "),
                        thinking_chunk("ing"),
                        content_chunk("42"),
                    ]
                )

        agent.llm = FakeLLM()
        orig = formatter.print_thinking_panel
        formatter.print_thinking_panel = lambda *a, **kw: None
        try:
            agent.call_llm(
                "task",
                streaming_callback=events.append,
                current_loop=1,
            )
        finally:
            formatter.print_thinking_panel = orig

        types = [e["type"] for e in events]
        assert "thinking_start" in types
        assert "thinking" in types
        assert "thinking_end" in types
        assert "content_start" in types
        assert "content" in types
        assert "content_end" in types
        assert types.index("thinking_start") < types.index(
            "content_start"
        )


@pytest.mark.skipif(
    not __import__("os").environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY required for live thinking stream test",
)
class TestThinkingStreamLive:
    MODEL = "claude-sonnet-4-5"

    def test_run_stream_with_events_includes_thinking(self):
        agent = Agent(
            agent_name="Reasoner",
            model_name=self.MODEL,
            thinking_tokens=1024,
            max_loops=1,
            print_on=False,
        )
        events = list(
            agent.run_stream(
                "Think step by step before answering: what is 17*23? "
                "Reply with just the number.",
                with_events=True,
            )
        )
        types = {e["type"] for e in events if isinstance(e, dict)}
        assert "content" in types
        assert "thinking" in types, (
            "thinking_tokens should enable extended thinking without "
            "reasoning_enabled=True"
        )
        assert "thinking_start" in types

    @pytest.mark.asyncio
    async def test_arun_stream_with_events(self):
        agent = Agent(
            agent_name="Reasoner",
            model_name=self.MODEL,
            thinking_tokens=1024,
            max_loops=1,
            print_on=False,
        )
        events = []
        async for evt in agent.arun_stream(
            "Think step by step: what is 9+9? Reply with just the number.",
            with_events=True,
        ):
            events.append(evt)
        types = {e.get("type") for e in events if isinstance(e, dict)}
        assert "content" in types
        assert "thinking" in types
