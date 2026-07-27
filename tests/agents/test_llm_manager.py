"""
Test suite for :class:`swarms.agents.llm_manager.LLMManager`.

Approach — no mocking framework for the LLM itself, real fakes instead:

* Every test constructs a real :class:`swarms.Agent` (offline: no network
  call happens during ``Agent.__init__`` or during ``LiteLLM.__init__``, so
  this is safe and fast) and then swaps ``agent.llm`` for a hand-written
  ``FakeLLM`` that mimics the two surfaces ``LLMManager`` actually touches:
  a ``.run(task=None, img=None, **kwargs)`` method and ``.stream`` /
  ``.temperature`` attributes.
* Streaming tests feed ``FakeLLM.run`` an iterator of ``FakeChunk`` objects
  that mimic litellm's ``ModelResponseStream`` shape closely enough for
  ``LLMManager`` (``chunk.choices[0].delta.content``,
  ``.delta.reasoning_content``, ``.delta.tool_calls[i].function.name`` /
  ``.arguments``, ``.finish_reason``, ``.logprobs``).
* Methods that rebuild the LLM (``switch_to_next_model``,
  ``reset_model_index``) call ``LLMManager.build()``, which constructs a
  real ``LiteLLM`` instance. That is safe offline — ``LiteLLM.__init__``
  makes no network call — so those tests let it happen and assert against
  the real instance's attributes (``model_name``, ``parallel_tool_calls``,
  ``init_kwargs``, ...).
* Fallback-chain tests monkeypatch ``agent.run`` (not ``agent.llm``) because
  ``handle_fallback_execution`` recurses through ``agent.run(...)``, not
  through the LLM directly.

No test makes a network call or requires an API key. Run with:

    cd swarms && PYTHONPATH=. python3 -m pytest tests/agents/test_llm_manager.py -v
    cd swarms && PYTHONPATH=. python3 -m pytest tests/agents/test_llm_manager.py -q
"""

from unittest.mock import Mock

import pytest

from swarms import Agent
from swarms.utils.litellm_wrapper import LiteLLM

########################################################
# Fakes: litellm-shaped stream chunks and a fake LLM
########################################################


class FakeFunction:
    """Mimics litellm's ``delta.tool_calls[i].function``."""

    def __init__(self, name: str = "", arguments: str = ""):
        self.name = name
        self.arguments = arguments


class FakeToolCallDelta:
    """Mimics one fragment of ``delta.tool_calls``."""

    def __init__(
        self,
        index: int,
        id: str = "",
        name: str = "",
        arguments: str = "",
    ):
        self.index = index
        self.id = id
        self.function = FakeFunction(name, arguments)


class FakeDelta:
    def __init__(
        self,
        content=None,
        reasoning_content=None,
        tool_calls=None,
    ):
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, delta, finish_reason=None, logprobs=None):
        self.delta = delta
        self.finish_reason = finish_reason
        self.logprobs = logprobs


class FakeChunk:
    """Mimics litellm's ``ModelResponseStream``."""

    def __init__(
        self,
        choices=None,
        usage=None,
        model=None,
        id=None,
        created=None,
    ):
        self.choices = choices or []
        self.usage = usage
        self.model = model
        self.id = id
        self.created = created


class FakeNoChoices:
    """A chunk with no ``choices`` attribute at all."""


class FakeLLM:
    """Stands in for ``agent.llm``.

    ``run()`` returns ``stream_return`` while ``self.stream`` is True (the
    manager flips this on before calling ``run``), otherwise
    ``non_stream_return``. Every call is recorded in ``.calls`` so tests can
    assert on forwarded args/kwargs.
    """

    def __init__(
        self,
        non_stream_return="ok",
        stream_return=None,
        temperature: float = 0.5,
    ):
        self.stream = False
        self.temperature = temperature
        self.non_stream_return = non_stream_return
        self.stream_return = (
            stream_return if stream_return is not None else []
        )
        self.calls = []

    def run(self, task=None, img=None, **kwargs):
        self.calls.append(
            {"task": task, "img": img, "kwargs": kwargs}
        )
        if self.stream:
            # Real litellm streams are single-pass generators, not
            # re-iterable lists — wrap so tests exercise the same
            # single-pass semantics the manager is written against.
            if isinstance(self.stream_return, (list, tuple)):
                return iter(self.stream_return)
            return self.stream_return
        return self.non_stream_return


class RaisingRunLLM:
    """``run()`` always raises, to exercise the ``call()`` error path."""

    def __init__(self, error: Exception):
        self.stream = False
        self.temperature = 0.5
        self.error = error

    def run(self, task=None, img=None, **kwargs):
        raise self.error


def content_chunk(text, finish_reason=None):
    return FakeChunk(
        choices=[FakeChoice(FakeDelta(content=text), finish_reason)]
    )


def reasoning_chunk(text):
    return FakeChunk(
        choices=[FakeChoice(FakeDelta(reasoning_content=text))]
    )


########################################################
# Fixtures
########################################################


@pytest.fixture
def agent():
    """A real, offline Agent — no fallback models configured."""
    return Agent(
        agent_name="TestAgent",
        model_name="gpt-4o-mini",
        persistent_memory=False,
        print_on=False,
        max_loops=1,
    )


@pytest.fixture
def fake_llm(agent):
    """Attach a bare FakeLLM to the fixture agent and return it."""
    llm = FakeLLM()
    agent.llm = llm
    return llm


########################################################
# Model selection & fallback rotation
########################################################


class TestGetAvailableModels:
    def test_uses_fallback_models_list_verbatim(self, agent):
        agent.fallback_models = ["m1", "m2", "m3"]
        result = agent.llm_manager.get_available_models()
        assert result == ["m1", "m2", "m3"]
        # It's a copy, not the same list object.
        assert result is not agent.fallback_models

    def test_builds_from_model_name_and_fallback_name(self, agent):
        agent.fallback_models = []
        agent.model_name = "primary"
        agent.fallback_model_name = "secondary"
        assert agent.llm_manager.get_available_models() == [
            "primary",
            "secondary",
        ]

    def test_deduplicates_identical_fallback_name(self, agent):
        agent.fallback_models = []
        agent.model_name = "same-model"
        agent.fallback_model_name = "same-model"
        assert agent.llm_manager.get_available_models() == [
            "same-model"
        ]

    def test_no_model_name_only_fallback(self, agent):
        agent.fallback_models = []
        agent.model_name = None
        agent.fallback_model_name = "only-fallback"
        assert agent.llm_manager.get_available_models() == [
            "only-fallback"
        ]

    def test_nothing_configured_returns_empty(self, agent):
        agent.fallback_models = []
        agent.model_name = None
        agent.fallback_model_name = None
        assert agent.llm_manager.get_available_models() == []


class TestGetCurrentModel:
    def test_returns_model_at_index(self, agent):
        agent.fallback_models = ["a", "b"]
        agent.current_model_index = 0
        assert agent.llm_manager.get_current_model() == "a"
        agent.current_model_index = 1
        assert agent.llm_manager.get_current_model() == "b"

    def test_out_of_range_falls_back_to_first_model(self, agent):
        agent.fallback_models = ["a", "b"]
        agent.current_model_index = 5
        assert agent.llm_manager.get_current_model() == "a"

    def test_out_of_range_with_no_models_uses_default(self, agent):
        agent.fallback_models = []
        agent.model_name = None
        agent.fallback_model_name = None
        agent.current_model_index = 0
        assert agent.llm_manager.get_current_model() == "gpt-5.4"


class TestSwitchToNextModel:
    def test_advances_index_updates_model_and_rebuilds_llm(
        self, agent, fake_llm
    ):
        agent.fallback_models = ["gpt-4o-mini", "gpt-4o"]
        agent.current_model_index = 0
        agent.model_name = "gpt-4o-mini"

        result = agent.llm_manager.switch_to_next_model()

        assert result is True
        assert agent.current_model_index == 1
        assert agent.model_name == "gpt-4o"
        # llm was rebuilt into a real LiteLLM instance, no longer the fake.
        assert isinstance(agent.llm, LiteLLM)
        assert agent.llm.model_name == "gpt-4o"

    def test_returns_false_when_models_are_exhausted(
        self, agent, fake_llm
    ):
        agent.fallback_models = ["gpt-4o-mini", "gpt-4o"]
        agent.current_model_index = 1
        agent.model_name = "gpt-4o"

        result = agent.llm_manager.switch_to_next_model()

        assert result is False
        # Nothing changed.
        assert agent.current_model_index == 1
        assert agent.model_name == "gpt-4o"
        assert agent.llm is fake_llm


class TestResetModelIndex:
    def test_returns_to_primary_model(self, agent, fake_llm):
        agent.fallback_models = ["gpt-4o-mini", "gpt-4o"]
        agent.current_model_index = 1
        agent.model_name = "gpt-4o"

        agent.llm_manager.reset_model_index()

        assert agent.current_model_index == 0
        assert agent.model_name == "gpt-4o-mini"
        assert isinstance(agent.llm, LiteLLM)
        assert agent.llm.model_name == "gpt-4o-mini"


class TestIsFallbackAvailable:
    def test_true_with_multiple_models(self, agent):
        agent.fallback_models = ["a", "b"]
        assert agent.llm_manager.is_fallback_available() is True

    def test_false_with_a_single_model(self, agent):
        agent.fallback_models = []
        agent.model_name = "only-one"
        agent.fallback_model_name = None
        assert agent.llm_manager.is_fallback_available() is False


########################################################
# Construction: build()
########################################################


class TestBuild:
    def test_returns_a_litellm_instance(self, agent):
        result = agent.llm_manager.build()
        assert isinstance(result, LiteLLM)

    def test_none_model_name_defaults_to_gpt_5_4(self, agent):
        agent.model_name = None
        result = agent.llm_manager.build()
        assert agent.model_name == "gpt-5.4"
        assert result.model_name == "gpt-5.4"

    def test_kwargs_and_single_dict_positional_arg_merge(self, agent):
        result = agent.llm_manager.build(
            {"custom_key": "val"}, foo="bar"
        )
        assert result.init_kwargs == {
            "foo": "bar",
            "custom_key": "val",
        }

    def test_non_dict_positional_args_land_under_additional_args(
        self, agent
    ):
        result = agent.llm_manager.build("pos1", "pos2")
        assert result.init_kwargs == {
            "additional_args": ("pos1", "pos2")
        }

    def test_llm_args_are_merged_in(self, agent):
        agent.llm_args = {"custom": 123}
        result = agent.llm_manager.build()
        assert result.init_kwargs == {"custom": 123}

    def test_parallel_tool_calls_true_with_two_or_more_tools(
        self, agent
    ):
        def f1(x: int) -> int:
            """doc"""
            return x

        def f2(x: int) -> int:
            """doc"""
            return x

        agent.tools = [f1, f2]
        agent.tools_list_dictionary = [
            {"type": "function", "function": {"name": "f1"}},
            {"type": "function", "function": {"name": "f2"}},
        ]
        result = agent.llm_manager.build()
        assert result.parallel_tool_calls is True
        assert result.tools_list_dictionary == (
            agent.tools_list_dictionary
        )

    def test_parallel_tool_calls_false_with_one_tool(self, agent):
        def f1(x: int) -> int:
            """doc"""
            return x

        agent.tools = [f1]
        agent.tools_list_dictionary = [
            {"type": "function", "function": {"name": "f1"}}
        ]
        result = agent.llm_manager.build()
        assert result.parallel_tool_calls is False

    def test_no_tools_list_omits_tool_keys(self, agent):
        agent.tools = None
        agent.tools_list_dictionary = []
        result = agent.llm_manager.build()
        assert result.tools_list_dictionary is None


########################################################
# Capability checks
########################################################


class TestCheckModelSupportsUtilities:
    def test_returns_none_and_never_raises_for_unsupported_model(
        self, agent
    ):
        agent.model_name = "not-a-real-model-xyz"
        agent.tools_list_dictionary = [
            {"type": "function", "function": {"name": "f"}}
        ]
        agent.tools = [lambda x: x, lambda y: y, lambda z: z]

        result = agent.llm_manager.check_model_supports_utilities(
            img="data:image/png;base64,AAAA"
        )

        assert result is None

    def test_no_image_no_tools_is_a_silent_noop(self, agent):
        agent.tools_list_dictionary = None
        agent.tools = None
        assert (
            agent.llm_manager.check_model_supports_utilities() is None
        )


########################################################
# randomize_temperature()
########################################################


class TestRandomizeTemperature:
    def test_sets_temperature_within_zero_one_range(
        self, agent, fake_llm
    ):
        for _ in range(20):
            agent.llm_manager.randomize_temperature()
            assert 0.0 <= agent.llm.temperature <= 1.0

    def test_falls_back_to_half_when_llm_has_no_temperature(
        self, agent
    ):
        class NoTemperature:
            pass

        agent.llm = NoTemperature()
        agent.llm_manager.randomize_temperature()
        assert agent.llm.temperature == 0.5

    def test_swallows_errors_from_the_setter(self, agent):
        class RaisingTemperature:
            @property
            def temperature(self):
                return 0.5

            @temperature.setter
            def temperature(self, value):
                raise RuntimeError("cannot set")

        agent.llm = RaisingTemperature()
        # Must not raise.
        agent.llm_manager.randomize_temperature()


########################################################
# stream_with_tool_collection()
########################################################


class TestStreamWithToolCollection:
    def test_forwards_every_chunk_unchanged(self, agent):
        chunks = [content_chunk("a"), content_chunk("b")]
        out = []
        collected = []
        for c in agent.llm_manager.stream_with_tool_collection(
            iter(chunks), out
        ):
            collected.append(c)
        assert collected == chunks
        assert out == []

    def test_assembles_fragmented_name_and_arguments(self, agent):
        chunks = [
            FakeChunk(
                choices=[
                    FakeChoice(
                        FakeDelta(
                            tool_calls=[
                                FakeToolCallDelta(
                                    0, id="call_1", name="get_"
                                )
                            ]
                        )
                    )
                ]
            ),
            FakeChunk(
                choices=[
                    FakeChoice(
                        FakeDelta(
                            tool_calls=[
                                FakeToolCallDelta(
                                    0,
                                    name="weather",
                                    arguments='{"city":',
                                )
                            ]
                        )
                    )
                ]
            ),
            FakeChunk(
                choices=[
                    FakeChoice(
                        FakeDelta(
                            tool_calls=[
                                FakeToolCallDelta(
                                    0, arguments='"NYC"}'
                                )
                            ]
                        )
                    )
                ]
            ),
        ]
        out = []
        list(
            agent.llm_manager.stream_with_tool_collection(
                iter(chunks), out
            )
        )
        assert out == [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city":"NYC"}',
                },
            }
        ]

    def test_handles_multiple_concurrent_indices(self, agent):
        chunks = [
            FakeChunk(
                choices=[
                    FakeChoice(
                        FakeDelta(
                            tool_calls=[
                                FakeToolCallDelta(
                                    0, id="c0", name="add"
                                ),
                                FakeToolCallDelta(
                                    1, id="c1", name="sub"
                                ),
                            ]
                        )
                    )
                ]
            ),
            FakeChunk(
                choices=[
                    FakeChoice(
                        FakeDelta(
                            tool_calls=[
                                FakeToolCallDelta(0, arguments="{}"),
                                FakeToolCallDelta(1, arguments="{}"),
                            ]
                        )
                    )
                ]
            ),
        ]
        out = []
        list(
            agent.llm_manager.stream_with_tool_collection(
                iter(chunks), out
            )
        )
        assert [c["function"]["name"] for c in out] == [
            "add",
            "sub",
        ]
        assert [c["id"] for c in out] == ["c0", "c1"]

    def test_no_tool_calls_leaves_output_empty(self, agent):
        chunks = [content_chunk("hello"), content_chunk("world")]
        out = []
        list(
            agent.llm_manager.stream_with_tool_collection(
                iter(chunks), out
            )
        )
        assert out == []


########################################################
# extract_thinking_from_stream()
########################################################


class TestExtractThinkingFromStream:
    def test_swallows_reasoning_yields_only_content(self, agent):
        chunks = [
            reasoning_chunk("let me think"),
            content_chunk("answer"),
        ]
        result = list(
            agent.llm_manager.extract_thinking_from_stream(
                iter(chunks)
            )
        )
        assert len(result) == 1
        assert result[0].choices[0].delta.content == "answer"

    def test_all_reasoning_no_content_yields_nothing(self, agent):
        chunks = [
            reasoning_chunk("thinking one"),
            reasoning_chunk("thinking two"),
        ]
        result = list(
            agent.llm_manager.extract_thinking_from_stream(
                iter(chunks)
            )
        )
        assert result == []

    def test_passes_through_chunks_with_no_choices(self, agent):
        marker = FakeNoChoices()
        chunks = [marker, content_chunk("hi")]
        result = list(
            agent.llm_manager.extract_thinking_from_stream(
                iter(chunks)
            )
        )
        assert result[0] is marker
        assert result[1].choices[0].delta.content == "hi"


########################################################
# call(): non-streaming path
########################################################


class TestCallNonStreaming:
    def test_returns_the_plain_string(self, agent, fake_llm):
        fake_llm.non_stream_return = "the answer"
        result = agent.llm_manager.call("what is python?")
        assert result == "the answer"

    def test_img_is_forwarded_to_llm_run(self, agent, fake_llm):
        agent.llm_manager.call("describe this", img="chart.png")
        assert fake_llm.calls[-1]["img"] == "chart.png"

    def test_is_last_is_stripped_from_kwargs(self, agent, fake_llm):
        agent.llm_manager.call("task", is_last=True, other_kwarg="x")
        forwarded = fake_llm.calls[-1]["kwargs"]
        assert "is_last" not in forwarded
        assert forwarded["other_kwarg"] == "x"

    def test_errors_are_logged_and_reraised(self, agent):
        agent.llm = RaisingRunLLM(ValueError("boom"))
        with pytest.raises(ValueError, match="boom"):
            agent.llm_manager.call("task")


########################################################
# call(): detailed streaming (agent.stream = True)
########################################################


class TestCallDetailedStreaming:
    def test_callback_invoked_once_per_token_with_token_info(
        self, agent, fake_llm
    ):
        fake_llm.stream_return = [
            content_chunk("Hello "),
            content_chunk("world", finish_reason="stop"),
        ]
        agent.stream = True
        tokens = []
        result = agent.llm_manager.call(
            "hi",
            current_loop=1,
            streaming_callback=lambda t: tokens.append(t),
        )

        assert result == "Hello world"
        assert len(tokens) == 2
        assert tokens[0]["token"] == "Hello "
        assert tokens[0]["token_index"] == 1
        assert tokens[1]["token"] == "world"
        assert tokens[1]["token_index"] == 2
        assert "timestamp" in tokens[0]

    def test_restores_llm_stream_flag_afterward(
        self, agent, fake_llm
    ):
        fake_llm.stream_return = [content_chunk("hi")]
        fake_llm.stream = False
        agent.stream = True

        agent.llm_manager.call("hi", current_loop=0)

        assert fake_llm.stream is False

    def test_tool_calls_returned_instead_of_text(
        self, agent, fake_llm
    ):
        fake_llm.stream_return = [
            FakeChunk(
                choices=[
                    FakeChoice(
                        FakeDelta(
                            tool_calls=[
                                FakeToolCallDelta(
                                    0, id="call_1", name="get_"
                                )
                            ]
                        )
                    )
                ]
            ),
            FakeChunk(
                choices=[
                    FakeChoice(
                        FakeDelta(
                            tool_calls=[
                                FakeToolCallDelta(
                                    0,
                                    name="weather",
                                    arguments='{"city":"NYC"}',
                                )
                            ]
                        ),
                        finish_reason="tool_calls",
                    )
                ]
            ),
        ]
        agent.stream = True
        tokens = []
        result = agent.llm_manager.call(
            "hi",
            current_loop=0,
            streaming_callback=lambda t: tokens.append(t),
        )

        assert result == [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city":"NYC"}',
                },
            }
        ]
        # No content chunks -> the callback never fired.
        assert tokens == []

    def test_non_iterable_string_response_falls_through(
        self, agent, fake_llm
    ):
        fake_llm.stream_return = "plain string result"
        agent.stream = True

        result = agent.llm_manager.call("hi", current_loop=0)

        assert result == "plain string result"
        assert fake_llm.stream is False


########################################################
# call(): panel streaming (agent.streaming_on = True)
########################################################


class TestCallPanelStreaming:
    def _chunks(self):
        return [
            reasoning_chunk("hmm let me think"),
            content_chunk("A"),
            content_chunk("B", finish_reason="stop"),
        ]

    def test_callback_path_forwards_content_and_swallows_thinking(
        self, agent, fake_llm
    ):
        fake_llm.stream_return = self._chunks()
        agent.streaming_on = True
        agent.print_on = False
        tokens = []

        result = agent.llm_manager.call(
            "hi",
            current_loop=0,
            streaming_callback=lambda t: tokens.append(t),
        )

        assert result == "AB"
        assert tokens == ["A", "B"]

    def test_silent_path_when_print_on_false(self, agent, fake_llm):
        fake_llm.stream_return = self._chunks()
        agent.streaming_on = True
        agent.print_on = False

        result = agent.llm_manager.call("hi", current_loop=0)

        assert result == "AB"

    def test_panel_path_when_print_on_true(self, agent, fake_llm):
        fake_llm.stream_return = self._chunks()
        agent.streaming_on = True
        agent.print_on = True

        result = agent.llm_manager.call("hi", current_loop=0)

        assert result == "AB"

    def test_tool_calls_returned_instead_of_text(
        self, agent, fake_llm
    ):
        fake_llm.stream_return = [
            FakeChunk(
                choices=[
                    FakeChoice(
                        FakeDelta(
                            tool_calls=[
                                FakeToolCallDelta(
                                    0, id="call_9", name="do_"
                                )
                            ]
                        )
                    )
                ]
            ),
            FakeChunk(
                choices=[
                    FakeChoice(
                        FakeDelta(
                            tool_calls=[
                                FakeToolCallDelta(
                                    0,
                                    name="thing",
                                    arguments="{}",
                                )
                            ]
                        ),
                        finish_reason="tool_calls",
                    )
                ]
            ),
        ]
        agent.streaming_on = True
        agent.print_on = False

        result = agent.llm_manager.call("hi", current_loop=0)

        assert result == [
            {
                "id": "call_9",
                "type": "function",
                "function": {
                    "name": "do_thing",
                    "arguments": "{}",
                },
            }
        ]

    def test_non_iterable_string_response_falls_through(
        self, agent, fake_llm
    ):
        fake_llm.stream_return = "plain string result"
        agent.streaming_on = True
        agent.print_on = False

        result = agent.llm_manager.call("hi", current_loop=0)

        assert result == "plain string result"
        assert fake_llm.stream is False


########################################################
# handle_fallback_execution()
########################################################


class TestHandleFallbackExecution:
    def test_no_fallbacks_calls_handle_run_error_and_returns_none(
        self, agent
    ):
        agent.fallback_models = []
        agent.fallback_model_name = None
        agent._handle_run_error = Mock()
        original_error = RuntimeError("boom")

        result = agent.llm_manager.handle_fallback_execution(
            task="t", original_error=original_error
        )

        assert result is None
        agent._handle_run_error.assert_called_once_with(
            original_error
        )

    def test_switches_model_and_reruns_on_success(self, agent):
        agent.fallback_model_name = "gpt-4o"
        agent.run = Mock(return_value="fallback result")

        result = agent.llm_manager.handle_fallback_execution(
            task="t", original_error=RuntimeError("boom")
        )

        assert result == "fallback result"
        assert agent.model_name == "gpt-4o"
        assert agent.current_model_index == 1
        agent.run.assert_called_once()

    def test_recurses_to_next_model_when_fallback_also_fails(
        self, agent
    ):
        agent.fallback_models = ["m1", "m2", "m3"]
        agent.model_name = "m1"
        agent.run = Mock(
            side_effect=[Exception("fallback 1 also failed"), "ok"]
        )

        result = agent.llm_manager.handle_fallback_execution(
            task="t", original_error=RuntimeError("boom")
        )

        assert result == "ok"
        assert agent.model_name == "m3"
        assert agent.current_model_index == 2
        assert agent.run.call_count == 2

    def test_exhausted_after_no_switch_calls_handle_run_error(
        self, agent
    ):
        agent.fallback_models = ["m1", "m2"]
        agent.model_name = "m2"
        agent.current_model_index = 1  # already at the last model
        agent._handle_run_error = Mock()
        original_error = RuntimeError("boom")

        result = agent.llm_manager.handle_fallback_execution(
            task="t", original_error=original_error
        )

        assert result is None
        agent._handle_run_error.assert_called_once_with(
            original_error
        )


########################################################
# Agent.* thin wrapper delegation
########################################################


class TestAgentWrapperDelegation:
    def test_get_available_models_delegates(self, agent):
        agent.llm_manager.get_available_models = Mock(
            return_value=["x"]
        )
        assert agent.get_available_models() == ["x"]
        agent.llm_manager.get_available_models.assert_called_once()

    def test_get_current_model_delegates(self, agent):
        agent.llm_manager.get_current_model = Mock(
            return_value="model-x"
        )
        assert agent.get_current_model() == "model-x"

    def test_switch_to_next_model_delegates(self, agent):
        agent.llm_manager.switch_to_next_model = Mock(
            return_value=True
        )
        assert agent.switch_to_next_model() is True

    def test_reset_model_index_delegates(self, agent):
        agent.llm_manager.reset_model_index = Mock()
        agent.reset_model_index()
        agent.llm_manager.reset_model_index.assert_called_once()

    def test_is_fallback_available_delegates(self, agent):
        agent.llm_manager.is_fallback_available = Mock(
            return_value=True
        )
        assert agent.is_fallback_available() is True

    def test_check_model_supports_utilities_delegates(self, agent):
        agent.llm_manager.check_model_supports_utilities = Mock(
            return_value=None
        )
        agent.check_model_supports_utilities(img="x.png")
        agent.llm_manager.check_model_supports_utilities.assert_called_once_with(
            img="x.png"
        )

    def test_dynamic_temperature_delegates(self, agent):
        agent.llm_manager.randomize_temperature = Mock()
        agent.dynamic_temperature()
        agent.llm_manager.randomize_temperature.assert_called_once()

    def test_get_llm_parameters_delegates(self, agent):
        agent.llm_manager.get_parameters = Mock(return_value="params")
        assert agent.get_llm_parameters() == "params"

    def test_call_llm_delegates(self, agent):
        agent.llm_manager.call = Mock(return_value="response")
        result = agent.call_llm(
            "task", img=None, current_loop=2, streaming_callback=None
        )
        assert result == "response"
        agent.llm_manager.call.assert_called_once_with(
            task="task",
            img=None,
            current_loop=2,
            streaming_callback=None,
        )

    def test_handle_fallback_execution_delegates(self, agent):
        agent.llm_manager.handle_fallback_execution = Mock(
            return_value="fb-result"
        )
        result = agent._handle_fallback_execution(
            task="t", original_error=ValueError("x")
        )
        assert result == "fb-result"
        agent.llm_manager.handle_fallback_execution.assert_called_once()

    def test_stream_with_tool_collection_delegates(self, agent):
        agent.llm_manager.stream_with_tool_collection = Mock(
            return_value=iter([])
        )
        out = []
        agent._stream_with_tool_collection(iter([]), out)
        agent.llm_manager.stream_with_tool_collection.assert_called_once()

    def test_extract_thinking_from_stream_delegates(self, agent):
        agent.llm_manager.extract_thinking_from_stream = Mock(
            return_value=iter([])
        )
        agent._extract_thinking_from_stream(iter([]))
        agent.llm_manager.extract_thinking_from_stream.assert_called_once()


########################################################
# get_parameters()
########################################################


class TestGetParameters:
    def test_returns_llm_vars_as_string(self, agent, fake_llm):
        fake_llm.temperature = 0.42
        result = agent.llm_manager.get_parameters()
        assert isinstance(result, str)
        assert "0.42" in result
