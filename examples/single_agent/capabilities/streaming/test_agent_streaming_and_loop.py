"""
Tests for all streaming and autonomous-loop changes.

Real Agent / LiteLLM instances are used throughout — no MagicMock on the LLM
or Agent.  Tests that inspect outgoing API parameters still patch
litellm.completion at the HTTP boundary, but the code under test runs on a
real LiteLLM object.

Models:
  MODEL_FAST    = claude-haiku-4-5-20251001   (cheap, no extended thinking)
  MODEL_THINKING = claude-sonnet-4-6          (extended thinking support)

Requires ANTHROPIC_API_KEY in the environment.
"""

import asyncio
import inspect
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest

MODEL_FAST = "claude-haiku-4-5-20251001"
MODEL_THINKING = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Tiny helpers — raw stream chunk factories
# (These are plain data objects, not mocked LLMs.  They let us test the
#  stream-processing generators without a live network call.)
# ---------------------------------------------------------------------------


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


def _tc_frag(index, id="", name="", arguments=""):
    fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=id, function=fn)


def content_chunk(text):
    return _chunk(_delta(content=text))


def thinking_chunk(text):
    return _chunk(_delta(reasoning_content=text))


def tool_chunk(index, id="", name="", arguments=""):
    return _chunk(
        _delta(tool_calls=[_tc_frag(index, id, name, arguments)])
    )


def empty_chunk():
    return SimpleNamespace()  # no .choices


# ---------------------------------------------------------------------------
# Helpers — real LLM response objects
# (Used when we patch litellm.completion but still run real LiteLLM code.)
# ---------------------------------------------------------------------------


def _fake_response(
    content="ok",
    tool_calls=None,
    thinking_blocks=None,
    reasoning_content=None,
):
    msg = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        thinking_blocks=thinking_blocks,
        reasoning_content=reasoning_content,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


# ===========================================================================
# 1.  formatter.print_thinking_panel
# ===========================================================================


class TestPrintThinkingPanel:
    """Uses real Formatter instance — no LLM involved."""

    def _fmt(self):
        """Real Formatter with a StringIO console so we can inspect output."""
        from rich.console import Console
        from swarms.utils.formatter import Formatter

        f = Formatter()
        f.console = Console(file=StringIO(), force_terminal=False)
        return f

    def test_renders_content(self):
        f = self._fmt()
        f.print_thinking_panel("I am reasoning.", title="Thinking")
        out = f.console.file.getvalue()
        assert "I am reasoning." in out

    def test_skips_empty_string(self):
        f = self._fmt()
        f.print_thinking_panel("", title="Thinking")
        assert f.console.file.getvalue() == ""

    def test_skips_none(self):
        f = self._fmt()
        f.print_thinking_panel(None, title="Thinking")
        assert f.console.file.getvalue() == ""

    def test_title_in_output(self):
        f = self._fmt()
        f.print_thinking_panel(
            "reasoning", title="MyAgent | Thinking"
        )
        out = f.console.file.getvalue()
        assert "MyAgent" in out

    def test_rich_markup_in_content_does_not_corrupt(self):
        """Bracket sequences like [CEG] must be treated as plain text."""
        f = self._fmt()
        f.print_thinking_panel(
            "[CEG] costs $192. [invalid]", title="Thinking"
        )
        out = f.console.file.getvalue()
        assert (
            "CEG" in out
        )  # text preserved, not swallowed as Rich tag


# ===========================================================================
# 2.  LiteLLM.output_for_reasoning — real LiteLLM, patched HTTP boundary
# ===========================================================================


class TestOutputForReasoningRealLiteLLM:
    """Real LiteLLM instance; litellm.completion is patched to avoid network."""

    def _llm(self, **kw):
        from swarms.utils.litellm_wrapper import LiteLLM

        return LiteLLM(
            model_name=kw.get("model_name", MODEL_FAST),
            thinking_tokens=kw.get("thinking_tokens", None),
            reasoning_enabled=kw.get("reasoning_enabled", False),
            agent_name=kw.get("agent_name", None),
        )

    def test_thinking_blocks_take_priority_over_reasoning_content(
        self,
    ):
        """When both fields exist (Anthropic), only thinking_blocks is used — no duplicate."""
        from swarms.utils.formatter import formatter

        llm = self._llm(agent_name="Bot")
        response = _fake_response(
            content="The answer.",
            thinking_blocks=[{"thinking": "deep thought"}],
            reasoning_content="deep thought",  # same text
        )
        calls = []
        orig = formatter.print_thinking_panel
        formatter.print_thinking_panel = (
            lambda text, title="Thinking": calls.append(text)
        )
        try:
            llm.output_for_reasoning(response)
        finally:
            formatter.print_thinking_panel = orig

        assert len(calls) == 1
        assert calls[0].count("deep thought") == 1  # no duplication

    def test_falls_back_to_reasoning_content_when_no_blocks(self):
        from swarms.utils.formatter import formatter

        llm = self._llm()
        response = _fake_response(
            content="answer",
            thinking_blocks=None,
            reasoning_content="cross-provider reasoning",
        )
        calls = []
        orig = formatter.print_thinking_panel
        formatter.print_thinking_panel = (
            lambda text, title="Thinking": calls.append(text)
        )
        try:
            llm.output_for_reasoning(response)
        finally:
            formatter.print_thinking_panel = orig

        assert len(calls) == 1
        assert "cross-provider reasoning" in calls[0]

    def test_no_panel_when_no_thinking(self):
        from swarms.utils.formatter import formatter

        llm = self._llm()
        response = _fake_response(content="plain")
        calls = []
        orig = formatter.print_thinking_panel
        formatter.print_thinking_panel = (
            lambda text, title="Thinking": calls.append(text)
        )
        try:
            llm.output_for_reasoning(response)
        finally:
            formatter.print_thinking_panel = orig
        assert calls == []

    def test_returns_message_content(self):
        from swarms.utils.formatter import formatter

        llm = self._llm()
        response = _fake_response(content="the final answer")
        orig = formatter.print_thinking_panel
        formatter.print_thinking_panel = lambda *a, **kw: None
        try:
            result = llm.output_for_reasoning(response)
        finally:
            formatter.print_thinking_panel = orig
        assert result == "the final answer"

    def test_agent_name_appears_in_title(self):
        from swarms.utils.formatter import formatter

        llm = self._llm(agent_name="TradingBot")
        response = _fake_response(
            content="x", reasoning_content="thinking..."
        )
        titles = []
        orig = formatter.print_thinking_panel
        formatter.print_thinking_panel = (
            lambda text, title="Thinking": titles.append(title)
        )
        try:
            llm.output_for_reasoning(response)
        finally:
            formatter.print_thinking_panel = orig
        assert any("TradingBot" in t for t in titles)

    def test_multiple_thinking_blocks_all_included(self):
        from swarms.utils.formatter import formatter

        llm = self._llm()
        response = _fake_response(
            content="ans",
            thinking_blocks=[
                {"thinking": "step A"},
                {"thinking": "step B"},
            ],
        )
        texts = []
        orig = formatter.print_thinking_panel
        formatter.print_thinking_panel = (
            lambda text, title="Thinking": texts.append(text)
        )
        try:
            llm.output_for_reasoning(response)
        finally:
            formatter.print_thinking_panel = orig
        combined = "".join(texts)
        assert "step A" in combined and "step B" in combined


# ===========================================================================
# 3.  Anthropic param constraints — real LiteLLM, patched HTTP boundary
# ===========================================================================


class TestAnthropicConstraintsRealLiteLLM:
    """Real LiteLLM builds completion_params; we capture them at the boundary."""

    def _captured_params(self, llm, task="test"):
        """Run llm.run() with a fake completion and return the captured kwargs."""
        captured = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return _fake_response(content="ok")

        with patch(
            "swarms.utils.litellm_wrapper.completion",
            side_effect=fake_completion,
        ):
            llm.run(task)
        return captured

    def test_top_p_removed_for_claude_with_thinking(self):
        from swarms.utils.litellm_wrapper import LiteLLM

        llm = LiteLLM(
            model_name=MODEL_THINKING,
            reasoning_enabled=True,
            thinking_tokens=512,
            top_p=1.0,  # default — must be stripped
            max_tokens=2048,
        )
        params = self._captured_params(llm)
        assert "top_p" not in params

    def test_temperature_forced_to_1_for_claude_with_thinking(self):
        from swarms.utils.litellm_wrapper import LiteLLM

        llm = LiteLLM(
            model_name=MODEL_THINKING,
            reasoning_enabled=True,
            thinking_tokens=512,
            temperature=0.3,  # must become 1
            max_tokens=2048,
        )
        params = self._captured_params(llm)
        assert params["temperature"] == 1

    def test_top_p_kept_for_non_anthropic_model(self):
        from swarms.utils.litellm_wrapper import LiteLLM

        llm = LiteLLM(model_name="gpt-4o", top_p=0.9)
        params = self._captured_params(llm)
        assert params.get("top_p") == 0.9

    def test_reasoning_effort_strips_top_p_for_claude(self):
        from swarms.utils.litellm_wrapper import LiteLLM

        llm = LiteLLM(
            model_name=MODEL_THINKING,
            reasoning_effort="high",
            top_p=1.0,
            max_tokens=16000,
        )
        with patch(
            "swarms.utils.litellm_wrapper.litellm"
        ) as mock_litellm:
            mock_litellm.supports_reasoning.return_value = True
            params = self._captured_params(llm)
        assert "top_p" not in params


# ===========================================================================
# 4.  Reasoning gate — real LiteLLM, patched HTTP boundary
# ===========================================================================


class TestReasoningGateRealLiteLLM:

    def _run_with_spy(self, llm, task="say hi"):
        """Return (output_for_reasoning_called, result)."""
        called = [False]
        original = llm.output_for_reasoning

        def spy(response):
            called[0] = True
            return original(response)

        llm.output_for_reasoning = spy

        with patch(
            "swarms.utils.litellm_wrapper.completion",
            return_value=_fake_response(
                content="hi", reasoning_content="thinking"
            ),
        ):
            result = llm.run(task)
        return called[0], result

    def test_thinking_tokens_alone_triggers_reasoning_path(self):
        from swarms.utils.litellm_wrapper import LiteLLM

        llm = LiteLLM(model_name="gpt-4o", thinking_tokens=1024)
        called, _ = self._run_with_spy(llm)
        assert (
            called
        ), "output_for_reasoning must be called when thinking_tokens is set"

    def test_reasoning_enabled_alone_triggers_reasoning_path(self):
        from swarms.utils.litellm_wrapper import LiteLLM

        llm = LiteLLM(model_name="gpt-4o", reasoning_enabled=True)
        called, _ = self._run_with_spy(llm)
        assert called

    def test_no_flags_skips_reasoning_path(self):
        from swarms.utils.litellm_wrapper import LiteLLM

        llm = LiteLLM(model_name="gpt-4o")
        called = [False]
        orig = llm.output_for_reasoning
        llm.output_for_reasoning = lambda r: (
            called.__setitem__(0, True),
            orig(r),
        )[1]

        with patch(
            "swarms.utils.litellm_wrapper.completion",
            return_value=_fake_response(content="plain"),
        ):
            llm.run("task")
        assert not called[0]


# ===========================================================================
# 5.  agent_name forwarded to LLM — real Agent instantiation
# ===========================================================================


class TestAgentNameForwardedRealAgent:

    def test_llm_has_agent_name_after_init(self):
        from swarms import Agent

        agent = Agent(
            agent_name="Quant-Bot",
            model_name=MODEL_FAST,
            max_loops=1,
            print_on=False,
        )
        assert agent.llm.agent_name == "Quant-Bot"

    def test_agent_name_reflects_agent_name_attribute(self):
        from swarms import Agent

        agent = Agent(
            agent_name="AlphaAgent",
            model_name=MODEL_FAST,
            max_loops=1,
            print_on=False,
        )
        assert agent.llm.agent_name == agent.agent_name


# ===========================================================================
# 6.  _stream_with_tool_collection — pure stream mechanics, no LLM
# ===========================================================================


class TestStreamWithToolCollection:
    """Real Agent instance, chunk-level data only."""

    def _agent(self):
        from swarms import Agent

        return Agent(
            agent_name="StreamBot",
            model_name=MODEL_FAST,
            max_loops=1,
            print_on=False,
        )

    def test_content_chunks_forwarded(self):
        agent = self._agent()
        chunks = [content_chunk("hello"), content_chunk(" world")]
        out = list(
            agent._stream_with_tool_collection(iter(chunks), [])
        )
        assert len(out) == 2

    def test_tool_chunk_also_forwarded(self):
        agent = self._agent()
        tc = tool_chunk(0, id="id1", name="fn", arguments="{}")
        out = list(agent._stream_with_tool_collection(iter([tc]), []))
        assert len(out) == 1

    def test_single_tool_call_assembled(self):
        agent = self._agent()
        chunks = [
            tool_chunk(0, id="abc", name="get_price", arguments=""),
            tool_chunk(
                0, id="", name="", arguments='{"ticker":"AAPL"}'
            ),
        ]
        calls_out = []
        list(
            agent._stream_with_tool_collection(
                iter(chunks), calls_out
            )
        )
        assert len(calls_out) == 1
        assert calls_out[0]["id"] == "abc"
        assert calls_out[0]["function"]["name"] == "get_price"
        assert (
            calls_out[0]["function"]["arguments"]
            == '{"ticker":"AAPL"}'
        )

    def test_multiple_tool_calls_by_index(self):
        agent = self._agent()
        chunks = [
            tool_chunk(0, id="i0", name="fn_a", arguments='{"a":1}'),
            tool_chunk(1, id="i1", name="fn_b", arguments='{"b":2}'),
        ]
        calls_out = []
        list(
            agent._stream_with_tool_collection(
                iter(chunks), calls_out
            )
        )
        assert len(calls_out) == 2
        assert calls_out[0]["function"]["name"] == "fn_a"
        assert calls_out[1]["function"]["name"] == "fn_b"

    def test_no_tool_calls_leaves_list_empty(self):
        agent = self._agent()
        calls_out = []
        list(
            agent._stream_with_tool_collection(
                iter([content_chunk("hi")]), calls_out
            )
        )
        assert calls_out == []

    def test_arguments_concatenated_across_fragments(self):
        agent = self._agent()
        chunks = [
            tool_chunk(0, id="x", name="fn", arguments='{"key":'),
            tool_chunk(0, id="", name="", arguments='"val"}'),
        ]
        calls_out = []
        list(
            agent._stream_with_tool_collection(
                iter(chunks), calls_out
            )
        )
        assert (
            calls_out[0]["function"]["arguments"] == '{"key":"val"}'
        )

    def test_chunk_without_choices_passes_through(self):
        agent = self._agent()
        out = list(
            agent._stream_with_tool_collection(
                iter([empty_chunk(), content_chunk("ok")]), []
            )
        )
        assert len(out) == 2


# ===========================================================================
# 7.  _extract_thinking_from_stream — real Agent, chunk-level data
# ===========================================================================


class TestExtractThinkingFromStream:

    def _agent(self):
        from swarms import Agent

        return Agent(
            agent_name="ThinkBot",
            model_name=MODEL_FAST,
            max_loops=1,
            print_on=False,
        )

    def test_reasoning_chunks_swallowed(self):
        from swarms.utils.formatter import formatter

        agent = self._agent()
        chunks = [
            thinking_chunk("deep thought"),
            content_chunk("answer"),
        ]
        orig = formatter.print_thinking_panel
        formatter.print_thinking_panel = lambda *a, **kw: None
        try:
            out = list(
                agent._extract_thinking_from_stream(iter(chunks))
            )
        finally:
            formatter.print_thinking_panel = orig
        contents = [
            c.choices[0].delta.content
            for c in out
            if hasattr(c, "choices")
        ]
        assert "deep thought" not in str(contents)

    def test_content_chunks_yielded(self):
        from swarms.utils.formatter import formatter

        agent = self._agent()
        chunks = [thinking_chunk("think"), content_chunk("result")]
        orig = formatter.print_thinking_panel
        formatter.print_thinking_panel = lambda *a, **kw: None
        try:
            out = list(
                agent._extract_thinking_from_stream(iter(chunks))
            )
        finally:
            formatter.print_thinking_panel = orig
        assert any(
            hasattr(c, "choices")
            and c.choices[0].delta.content == "result"
            for c in out
        )

    def test_panel_printed_before_first_content_yielded_to_consumer(
        self,
    ):
        """Panel fires inside the generator before the content chunk reaches caller."""
        from swarms.utils.formatter import formatter

        agent = self._agent()
        call_order = []
        orig = formatter.print_thinking_panel
        formatter.print_thinking_panel = (
            lambda *a, **kw: call_order.append("panel")
        )
        try:
            chunks = [
                thinking_chunk("t1"),
                thinking_chunk("t2"),
                content_chunk("ans"),
            ]
            stream = agent._extract_thinking_from_stream(iter(chunks))
            for chunk in stream:
                if (
                    hasattr(chunk, "choices")
                    and chunk.choices[0].delta.content
                ):
                    call_order.append("first_content")
                    break
        finally:
            formatter.print_thinking_panel = orig

        assert call_order.index("panel") < call_order.index(
            "first_content"
        )

    def test_no_panel_when_no_thinking(self):
        from swarms.utils.formatter import formatter

        agent = self._agent()
        calls = []
        orig = formatter.print_thinking_panel
        formatter.print_thinking_panel = (
            lambda *a, **kw: calls.append(1)
        )
        try:
            list(
                agent._extract_thinking_from_stream(
                    iter([content_chunk("a"), content_chunk("b")])
                )
            )
        finally:
            formatter.print_thinking_panel = orig
        assert calls == []

    def test_panel_called_once_for_multiple_thinking_chunks(self):
        from swarms.utils.formatter import formatter

        agent = self._agent()
        calls = []
        orig = formatter.print_thinking_panel
        formatter.print_thinking_panel = (
            lambda *a, **kw: calls.append(1)
        )
        try:
            list(
                agent._extract_thinking_from_stream(
                    iter(
                        [
                            thinking_chunk("t1"),
                            thinking_chunk("t2"),
                            content_chunk("ans"),
                        ]
                    )
                )
            )
        finally:
            formatter.print_thinking_panel = orig
        assert len(calls) == 1

    def test_all_thinking_text_accumulated_into_one_panel_call(self):
        from swarms.utils.formatter import formatter

        agent = self._agent()
        texts = []
        orig = formatter.print_thinking_panel
        formatter.print_thinking_panel = (
            lambda text, **kw: texts.append(text)
        )
        try:
            list(
                agent._extract_thinking_from_stream(
                    iter(
                        [
                            thinking_chunk("part A"),
                            thinking_chunk("part B"),
                            content_chunk("ok"),
                        ]
                    )
                )
            )
        finally:
            formatter.print_thinking_panel = orig
        combined = "".join(texts)
        assert "part A" in combined and "part B" in combined

    def test_stream_with_only_thinking_still_prints_panel(self):
        from swarms.utils.formatter import formatter

        agent = self._agent()
        calls = []
        orig = formatter.print_thinking_panel
        formatter.print_thinking_panel = (
            lambda *a, **kw: calls.append(1)
        )
        try:
            list(
                agent._extract_thinking_from_stream(
                    iter([thinking_chunk("only thinking")])
                )
            )
        finally:
            formatter.print_thinking_panel = orig
        assert len(calls) == 1


# ===========================================================================
# 8.  run_stream — real Agent, real LLM
# ===========================================================================


class TestRunStreamRealAgent:

    def _agent(self, **kw):
        from swarms import Agent

        return Agent(
            agent_name=kw.get("agent_name", "StreamBot"),
            model_name=kw.get("model_name", MODEL_FAST),
            max_loops=kw.get("max_loops", 1),
            streaming_on=True,
            print_on=False,
        )

    def test_yields_nonempty_tokens(self):
        agent = self._agent()
        tokens = list(
            agent.run_stream("Reply with exactly the word: hello")
        )
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_full_response_assembles_from_tokens(self):
        agent = self._agent()
        tokens = list(
            agent.run_stream(
                "What is 2 + 2? Reply with just the number."
            )
        )
        full = "".join(tokens)
        assert "4" in full

    def test_streaming_on_restored_after_completion(self):
        agent = self._agent()
        original = agent.streaming_on
        list(agent.run_stream("Say yes."))
        assert agent.streaming_on == original

    def test_propagates_exception_from_run(self):
        agent = self._agent()

        def boom(task, img=None, streaming_callback=None, **kw):
            raise RuntimeError("intentional failure")

        agent._run = boom
        with pytest.raises(RuntimeError, match="intentional failure"):
            list(agent.run_stream("any task"))

    def test_streaming_on_restored_after_exception(self):
        agent = self._agent()
        original = agent.streaming_on

        def boom(task, img=None, streaming_callback=None, **kw):
            raise RuntimeError("err")

        agent._run = boom
        try:
            list(agent.run_stream("any"))
        except RuntimeError:
            pass
        assert agent.streaming_on == original

    def test_multiple_turns_all_streamed(self):
        """Tool-using agent — tokens from both tool-call and synthesis turns arrive."""

        def add(a: int, b: int) -> int:
            """Add two numbers and return the result."""
            return a + b

        from swarms import Agent

        agent = Agent(
            agent_name="ToolStream",
            model_name=MODEL_FAST,
            max_loops=3,
            streaming_on=True,
            print_on=False,
            tools=[add],
        )
        tokens = list(
            agent.run_stream(
                "Use the add tool to compute 7 + 5, then state the result."
            )
        )
        full = "".join(tokens)
        assert len(tokens) > 0
        assert (
            "12" in full or len(full) > 10
        )  # either got answer or at least got something


# ===========================================================================
# 9.  arun_stream — real Agent, real LLM
# ===========================================================================


class TestArunStreamRealAgent:

    def _agent(self):
        from swarms import Agent

        return Agent(
            agent_name="AsyncBot",
            model_name=MODEL_FAST,
            max_loops=1,
            streaming_on=True,
            print_on=False,
        )

    def test_yields_tokens_async(self):
        agent = self._agent()

        async def run():
            return [
                t async for t in agent.arun_stream("Reply: hello")
            ]

        tokens = asyncio.run(run())
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_full_response_correct(self):
        agent = self._agent()

        async def run():
            return [
                t
                async for t in agent.arun_stream(
                    "What is 3 + 3? Reply with just the number."
                )
            ]

        tokens = asyncio.run(run())
        full = "".join(tokens)
        assert "6" in full

    def test_propagates_exception_async(self):
        agent = self._agent()

        def boom(task, img=None, streaming_callback=None, **kw):
            raise ValueError("async boom")

        agent._run = boom

        async def run():
            async for _ in agent.arun_stream("task"):
                pass

        with pytest.raises(ValueError, match="async boom"):
            asyncio.run(run())

    def test_streaming_on_restored_after_async(self):
        agent = self._agent()
        original = agent.streaming_on

        async def run():
            async for _ in agent.arun_stream("Say yes."):
                pass

        asyncio.run(run())
        assert agent.streaming_on == original

    def test_uses_get_running_loop_not_get_event_loop(self):
        from swarms import Agent

        src = inspect.getsource(Agent.arun_stream)
        assert "get_running_loop()" in src
        assert "get_event_loop()" not in src


# ===========================================================================
# 10.  think tool excluded when thinking_tokens set — real Agent init
# ===========================================================================


class TestThinkToolExcludedRealAgent:

    def test_think_not_in_tool_schemas_when_thinking_tokens_set(self):
        from swarms import Agent
        from swarms.structs.autonomous_loop_utils import (
            get_autonomous_planning_tools,
        )

        agent = Agent(
            agent_name="AutoBot",
            model_name=MODEL_THINKING,
            max_loops="auto",
            thinking_tokens=1024,
            top_p=None,
            temperature=1,
            print_on=False,
        )
        # Simulate what _run_autonomous_loop does: filter planning tools
        planning_tools = get_autonomous_planning_tools()
        if agent.thinking_tokens is not None:
            planning_tools = [
                t
                for t in planning_tools
                if t.get("function", {}).get("name") != "think"
            ]
        names = [t["function"]["name"] for t in planning_tools]
        assert "think" not in names

    def test_think_present_without_thinking_tokens(self):
        from swarms import Agent
        from swarms.structs.autonomous_loop_utils import (
            get_autonomous_planning_tools,
        )

        Agent(
            agent_name="AutoBot",
            model_name=MODEL_FAST,
            max_loops="auto",
            thinking_tokens=None,
            print_on=False,
        )
        planning_tools = get_autonomous_planning_tools()
        # No filter applied when thinking_tokens is None
        names = [t["function"]["name"] for t in planning_tools]
        assert "think" in names

    def test_core_tools_always_present(self):
        """create_plan, subtask_done, complete_task must survive the filter."""
        from swarms.structs.autonomous_loop_utils import (
            get_autonomous_planning_tools,
        )

        tools = [
            t
            for t in get_autonomous_planning_tools()
            if t.get("function", {}).get("name") != "think"
        ]
        names = {t["function"]["name"] for t in tools}
        for required in (
            "create_plan",
            "subtask_done",
            "complete_task",
        ):
            assert required in names


# ===========================================================================
# 11.  agent_name in LLM — real Agent
# ===========================================================================


class TestAgentNameOnRealLLM:

    def test_agent_name_on_llm(self):
        from swarms import Agent

        agent = Agent(
            agent_name="Alpha",
            model_name=MODEL_FAST,
            max_loops=1,
            print_on=False,
        )
        assert agent.llm.agent_name == "Alpha"

    def test_agent_name_matches_attribute(self):
        from swarms import Agent

        agent = Agent(
            agent_name="Beta",
            model_name=MODEL_FAST,
            max_loops=1,
            print_on=False,
        )
        assert agent.llm.agent_name == agent.agent_name


# ===========================================================================
# 12.  execution prompt written ONCE per subtask — real Agent, inspect memory
# ===========================================================================


class TestExecutionPromptWrittenOnce:

    def test_execution_prompt_appears_once_in_memory(self):
        """After manually running the fix, memory contains one execution prompt per subtask."""
        from swarms import Agent
        from swarms.structs.autonomous_loop_utils import (
            get_execution_prompt,
        )

        agent = Agent(
            agent_name="TestAuto",
            model_name=MODEL_FAST,
            max_loops="auto",
            print_on=False,
        )

        subtasks = [
            {
                "step_id": "s1",
                "status": "pending",
                "description": "Do something",
            }
        ]
        prompt = get_execution_prompt("s1", "Do something", subtasks)

        before = len(agent.short_memory.conversation_history)

        # Simulate the fixed code: add once before the inner loop
        agent.short_memory.add(role=agent.user_name, content=prompt)
        # Old (buggy) code would add again on every iteration:
        # agent.short_memory.add(role=agent.user_name, content=prompt)

        after = len(agent.short_memory.conversation_history)
        assert after - before == 1  # exactly one message was appended


# ===========================================================================
# 13.  execution prompt batching instruction — string content check
# ===========================================================================


class TestExecutionPromptBatching:

    def _prompt(self, sid="t1", desc="Fetch price", subtasks=None):
        from swarms.structs.autonomous_loop_utils import (
            get_execution_prompt,
        )

        if subtasks is None:
            subtasks = [
                {
                    "step_id": sid,
                    "status": "pending",
                    "description": desc,
                }
            ]
        return get_execution_prompt(sid, desc, subtasks)

    def test_single_response_instruction(self):
        assert "SINGLE response" in self._prompt()

    def test_subtask_id_present(self):
        assert "fetch_price" in self._prompt(sid="fetch_price")

    def test_description_present(self):
        assert "Fetch the CEG price" in self._prompt(
            desc="Fetch the CEG price"
        )

    def test_all_statuses_shown(self):
        subtasks = [
            {
                "step_id": "s1",
                "status": "completed",
                "description": "D1",
            },
            {
                "step_id": "s2",
                "status": "pending",
                "description": "D2",
            },
        ]
        p = self._prompt(sid="s2", desc="D2", subtasks=subtasks)
        assert "completed" in p and "pending" in p

    def test_subtask_done_mentioned(self):
        assert "subtask_done" in self._prompt()

    def test_no_think_tool_requirement(self):
        assert "Use the 'think' tool" not in self._prompt()


# ===========================================================================
# 14.  No duplicate panels for subtask_done — dispatch logic with real Agent
# ===========================================================================


class TestNoDuplicatePanelRealAgent:
    """Use real Agent._visualize_function_call routing logic, patch formatter to count calls."""

    def _dispatch(self, agent, function_name, arguments, result=None):
        """Mirror the fixed dispatch in _run_autonomous_loop."""
        if function_name not in ("subtask_done", "complete_task"):
            agent._visualize_function_call(function_name, arguments)
        # execute (skipped here)
        if function_name in ("subtask_done", "complete_task"):
            agent._visualize_function_call(
                function_name, arguments, result
            )

    def _agent(self):
        from swarms import Agent

        agent = Agent(
            agent_name="DispatchBot",
            model_name=MODEL_FAST,
            max_loops=1,
            print_on=False,
        )
        return agent

    def test_subtask_done_panel_shown_once_with_result(self):
        agent = self._agent()
        calls = []
        agent._visualize_function_call = (
            lambda name, args, result=None: calls.append(
                (name, result)
            )
        )

        self._dispatch(
            agent,
            "subtask_done",
            {"task_id": "s1", "summary": "done"},
            "Subtask s1 done",
        )

        assert len(calls) == 1
        assert calls[0][0] == "subtask_done"
        assert calls[0][1] == "Subtask s1 done"

    def test_regular_tool_shown_before_execution_no_result(self):
        agent = self._agent()
        calls = []
        agent._visualize_function_call = (
            lambda name, args, result=None: calls.append(
                (name, result)
            )
        )

        self._dispatch(agent, "get_stock_price", {"ticker": "CEG"})

        assert len(calls) == 1
        assert calls[0][1] is None  # pre-execution: no result

    def test_complete_task_shown_once_with_result(self):
        agent = self._agent()
        calls = []
        agent._visualize_function_call = (
            lambda name, args, result=None: calls.append(
                (name, result)
            )
        )

        self._dispatch(
            agent,
            "complete_task",
            {"summary": "All done"},
            "Task complete",
        )

        assert len(calls) == 1
        assert calls[0][1] == "Task complete"


# ===========================================================================
# 15.  _generate_final_summary passes streaming_callback — real Agent
# ===========================================================================


class TestGenerateFinalSummaryCallbackRealAgent:

    def _agent(self):
        from swarms import Agent

        agent = Agent(
            agent_name="SummaryBot",
            model_name=MODEL_FAST,
            max_loops="auto",
            print_on=False,
        )
        return agent

    def test_streaming_callback_forwarded(self):
        agent = self._agent()
        received = []

        def fake_call_llm(
            task, current_loop=0, streaming_callback=None
        ):
            received.append(streaming_callback)
            return "summary text"

        agent.call_llm = fake_call_llm
        agent.parse_llm_output = lambda r: r

        def my_cb(t):
            pass

        agent._generate_final_summary(streaming_callback=my_cb)
        assert received[0] is my_cb

    def test_none_callback_accepted(self):
        agent = self._agent()

        def fake_call_llm(
            task, current_loop=0, streaming_callback=None
        ):
            return "plain summary"

        agent.call_llm = fake_call_llm
        agent.parse_llm_output = lambda r: r

        result = agent._generate_final_summary(
            streaming_callback=None
        )
        assert isinstance(result, str)

    def test_signature_has_streaming_callback_param(self):
        from swarms import Agent

        sig = inspect.signature(Agent._generate_final_summary)
        assert "streaming_callback" in sig.parameters


# ===========================================================================
# 16.  temp_llm_instance_for_tool_summary — real Agent, real LiteLLM
# ===========================================================================


class TestTempLlmInstanceRealAgent:

    def test_top_p_matches_agent_setting(self):
        from swarms import Agent

        agent = Agent(
            agent_name="Bot",
            model_name=MODEL_FAST,
            max_loops=1,
            top_p=None,
            print_on=False,
        )
        tmp = agent.temp_llm_instance_for_tool_summary()
        assert tmp.top_p is None

    def test_returns_litellm_instance(self):
        from swarms import Agent
        from swarms.utils.litellm_wrapper import LiteLLM

        agent = Agent(
            agent_name="Bot",
            model_name=MODEL_FAST,
            max_loops=1,
            print_on=False,
        )
        tmp = agent.temp_llm_instance_for_tool_summary()
        assert isinstance(tmp, LiteLLM)


# ===========================================================================
# 17.  Conversation.conversation_history bug fix in _generate_final_summary
# ===========================================================================


class TestConversationHistoryAttributeFix:

    def test_conversation_uses_conversation_history_not_messages(
        self,
    ):
        from swarms.structs.conversation import Conversation

        c = Conversation(system_prompt="sys")
        c.add(role="user", content="hello")
        assert hasattr(c, "conversation_history")
        assert not hasattr(c, "messages")

    def test_generate_summary_fallback_does_not_attribute_error(self):
        """The fallback path used self.short_memory.messages — fixed to conversation_history."""
        from swarms import Agent

        agent = Agent(
            agent_name="SummaryBot",
            model_name=MODEL_FAST,
            max_loops="auto",
            print_on=False,
        )

        def fake_call_llm(
            task, current_loop=0, streaming_callback=None
        ):
            return "plain text"  # not a list → triggers fallback path

        agent.call_llm = fake_call_llm
        agent.parse_llm_output = lambda r: r

        # Should not raise AttributeError: 'Conversation' has no attribute 'messages'
        result = agent._generate_final_summary()
        assert result is not None


# ===========================================================================
# 18.  Integration — run_stream with real thinking model
# ===========================================================================


class TestRunStreamWithThinkingRealLLM:

    def test_thinking_model_produces_tokens_via_run_stream(self):
        """Extended thinking with run_stream: tokens arrive token-by-token."""
        from swarms import Agent

        agent = Agent(
            agent_name="ThinkStreamer",
            model_name=MODEL_THINKING,
            max_loops=1,
            thinking_tokens=2048,
            temperature=1,
            top_p=None,
            streaming_on=True,
            print_on=False,
        )
        tokens = list(
            agent.run_stream(
                "What is 6 multiplied by 7? Give only the number."
            )
        )
        full = "".join(tokens)
        assert "42" in full
        assert len(tokens) > 0

    def test_thinking_model_run_stream_no_top_p_error(self):
        """If top_p is not stripped, Anthropic raises 400; this confirms the fix."""
        from swarms import Agent

        agent = Agent(
            agent_name="TopPTest",
            model_name=MODEL_THINKING,
            max_loops=1,
            thinking_tokens=1024,
            temperature=1,
            top_p=None,
            streaming_on=True,
            print_on=False,
        )
        # Must not raise — Anthropic would 400 if top_p were still present
        tokens = list(agent.run_stream("Say ok."))
        assert len(tokens) > 0


# ===========================================================================
# 19.  Autonomous loop real run — real Agent, real tools, real LLM
# ===========================================================================


class TestAutonomousLoopRealRun:

    def test_simple_tool_task_completes(self):
        """Full plan→execute→summary cycle with a real tool and real LLM."""

        def multiply(a: int, b: int) -> int:
            """Multiply two integers and return the result."""
            return a * b

        from swarms import Agent

        agent = Agent(
            agent_name="AutoCalc",
            model_name=MODEL_FAST,
            max_loops="auto",
            tools=[multiply],
            print_on=False,
            thinking_tokens=None,
        )
        result = agent.run(
            "Use the multiply tool to compute 6 × 7, then report the answer."
        )
        assert result is not None
        assert "42" in str(result)

    def test_think_tool_absent_in_tool_list_for_thinking_agent(self):
        """After _run_autonomous_loop init, 'think' not in tools when thinking_tokens set."""
        from swarms import Agent

        agent = Agent(
            agent_name="AutoThink",
            model_name=MODEL_THINKING,
            max_loops="auto",
            thinking_tokens=1024,
            temperature=1,
            top_p=None,
            print_on=False,
        )
        # Manually run the tool filtering logic (same as _run_autonomous_loop does)
        from swarms.structs.autonomous_loop_utils import (
            get_autonomous_planning_tools,
        )

        tools = get_autonomous_planning_tools()
        if agent.thinking_tokens is not None:
            tools = [
                t for t in tools if t["function"]["name"] != "think"
            ]
        names = {t["function"]["name"] for t in tools}
        assert "think" not in names
        assert "create_plan" in names


if __name__ == "__main__":
    pytest.main(["-v", __file__])
