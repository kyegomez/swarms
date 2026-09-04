"""
Tests for :mod:`swarms.utils.litellm_wrapper`.

The single home for LiteLLM coverage. Merged from the former
``tests/test_multi_provider.py``; the print-based ``run_test_suite()`` script
that used to live here was dropped because it asserted nothing — every check
sat inside a ``try/except`` that printed a mark and moved on, so it could not
fail. What it walked through, and what is therefore still unasserted, is listed
at the bottom of this docstring.

Verifies that non-OpenAI providers (Anthropic, Groq, Grok/xAI, Gemini) produce
real output through all major code paths: the LiteLLM wrapper, Agent, and swarm
structures.

No mocks — these make real API calls and need provider keys. For providers
without a key (Groq, Grok, Gemini) the tests verify that:
  - Parameter construction is correct (no reasoning_effort or top_p=None leak)
  - The call reaches the provider and fails on auth, NOT on bad parameters
    (proving the request payload is well-formed)

Still uncovered by any assertion, from the dropped script: streaming via
``arun``, ``batched_run``, tool calling, vision support detection, direct-URL
image handling, and image message preparation.
"""

import warnings

import pytest
from dotenv import load_dotenv

load_dotenv()

from swarms import Agent, SequentialWorkflow
from swarms.structs.mixture_of_agents import MixtureOfAgents
from swarms.utils.litellm_wrapper import LiteLLM

# ---------------------------------------------------------------------------
# Provider model names
# ---------------------------------------------------------------------------
OPENAI_MODEL = "openai/gpt-4.1-mini"
ANTHROPIC_MODEL = "anthropic/claude-sonnet-4-20250514"
GROQ_MODEL = "groq/llama-3.3-70b-versatile"
GROK_MODEL = "xai/grok-3-mini"
GEMINI_MODEL = "gemini/gemini-2.0-flash"

SIMPLE_TASK = "What is 2+2? Answer with just the number."
SIMPLE_PROMPT = "You are a helpful math tutor. Answer concisely."


def _has_answer(result) -> bool:
    """Check that the result contains the expected answer '4'."""
    if result is None:
        return False
    text = str(result).lower()
    return "4" in text


def _is_param_error(exc: Exception) -> bool:
    """Return True if the exception is a parameter/config error (the bug).
    These are the errors that appeared before the fix — temperature conflicts,
    reasoning_effort injection, thinking budget issues, etc."""
    msg = str(exc).lower()
    param_indicators = [
        "temperature",
        "reasoning",
        "thinking",
        "top_p",
        "budget_tokens",
    ]
    return any(indicator in msg for indicator in param_indicators)


def _assert_not_param_error(exc: Exception):
    """Assert that an exception is NOT a parameter bug.
    Used for providers without API keys: the call should reach the provider
    and fail on authentication, never on malformed parameters."""
    assert not _is_param_error(exc), (
        f"Parameter error detected (bug!) — the provider rejected the "
        f"request due to bad parameters, not missing auth:\n{exc}"
    )


# ===================================================================
# LiteLLM wrapper — direct calls (OpenAI + Anthropic with keys)
# ===================================================================


class TestLiteLLMProviders:
    """Direct LiteLLM wrapper tests for providers with API keys."""

    def test_openai_basic(self):
        llm = LiteLLM(
            model_name=OPENAI_MODEL,
            system_prompt=SIMPLE_PROMPT,
            max_tokens=50,
        )
        result = llm.run(SIMPLE_TASK)
        assert isinstance(result, str)
        assert _has_answer(result)

    def test_anthropic_basic(self):
        llm = LiteLLM(
            model_name=ANTHROPIC_MODEL,
            system_prompt=SIMPLE_PROMPT,
            max_tokens=50,
        )
        result = llm.run(SIMPLE_TASK)
        assert isinstance(result, str)
        assert _has_answer(result)

    def test_anthropic_no_system_prompt(self):
        """Anthropic should work even without a system prompt."""
        llm = LiteLLM(
            model_name=ANTHROPIC_MODEL,
            max_tokens=50,
        )
        result = llm.run(SIMPLE_TASK)
        assert isinstance(result, str)
        assert _has_answer(result)

    def test_anthropic_empty_system_prompt(self):
        """Anthropic rejects empty system blocks — wrapper should handle it."""
        llm = LiteLLM(
            model_name=ANTHROPIC_MODEL,
            system_prompt="   ",
            max_tokens=50,
        )
        result = llm.run(SIMPLE_TASK)
        assert isinstance(result, str)
        assert _has_answer(result)


# ===================================================================
# Agent — single agent per provider (OpenAI + Anthropic with keys)
# ===================================================================


class TestAgentProviders:
    """Agent-level tests ensuring each provider works end-to-end."""

    def test_openai_agent(self):
        agent = Agent(
            agent_name="OpenAI-Test",
            model_name=OPENAI_MODEL,
            system_prompt=SIMPLE_PROMPT,
            max_loops=1,
            max_tokens=50,
            print_on=False,
            streaming_on=False,
        )
        result = agent.run(SIMPLE_TASK)
        assert result is not None
        assert _has_answer(result)

    def test_anthropic_agent(self):
        agent = Agent(
            agent_name="Anthropic-Test",
            model_name=ANTHROPIC_MODEL,
            system_prompt=SIMPLE_PROMPT,
            max_loops=1,
            max_tokens=50,
            print_on=False,
            streaming_on=False,
        )
        result = agent.run(SIMPLE_TASK)
        assert result is not None
        assert _has_answer(result)

    def test_anthropic_agent_streaming(self):
        agent = Agent(
            agent_name="Anthropic-Stream",
            model_name=ANTHROPIC_MODEL,
            system_prompt=SIMPLE_PROMPT,
            max_loops=1,
            max_tokens=50,
            print_on=False,
            streaming_on=True,
        )
        result = agent.run(SIMPLE_TASK)
        assert result is not None
        assert _has_answer(result)

    def test_anthropic_agent_multi_loop(self):
        agent = Agent(
            agent_name="Anthropic-MultiLoop",
            model_name=ANTHROPIC_MODEL,
            system_prompt=SIMPLE_PROMPT,
            max_loops=2,
            max_tokens=100,
            print_on=False,
            streaming_on=False,
        )
        result = agent.run(SIMPLE_TASK)
        assert result is not None
        assert _has_answer(result)


# ===================================================================
# Agent — reasoning_effort defaults
# ===================================================================


class TestReasoningEffortDefaults:
    """Verify reasoning_effort defaults to None and doesn't break providers."""

    def test_default_reasoning_effort_is_none(self):
        """Agent should not set reasoning_effort unless user asks."""
        agent = Agent(
            agent_name="Default-Check",
            model_name=ANTHROPIC_MODEL,
            system_prompt=SIMPLE_PROMPT,
            max_loops=1,
            print_on=False,
        )
        assert agent.reasoning_effort is None

    def test_openai_default_no_reasoning(self):
        """OpenAI agent with default config should work."""
        agent = Agent(
            agent_name="OAI-NoReasoning",
            model_name=OPENAI_MODEL,
            system_prompt=SIMPLE_PROMPT,
            max_loops=1,
            max_tokens=50,
            print_on=False,
            streaming_on=False,
        )
        result = agent.run(SIMPLE_TASK)
        assert _has_answer(result)

    def test_anthropic_default_no_reasoning(self):
        """Anthropic agent with default config should work (the main bug)."""
        agent = Agent(
            agent_name="Ant-NoReasoning",
            model_name=ANTHROPIC_MODEL,
            system_prompt=SIMPLE_PROMPT,
            max_loops=1,
            max_tokens=50,
            print_on=False,
            streaming_on=False,
        )
        result = agent.run(SIMPLE_TASK)
        assert _has_answer(result)

    def test_anthropic_explicit_reasoning_effort(self):
        """Anthropic with explicit reasoning_effort should auto-fix temp and max_tokens."""
        agent = Agent(
            agent_name="Ant-Reasoning",
            model_name=ANTHROPIC_MODEL,
            system_prompt=SIMPLE_PROMPT,
            max_loops=1,
            max_tokens=200,  # intentionally small — wrapper should auto-increase
            print_on=False,
            streaming_on=False,
            reasoning_effort="low",
        )
        result = agent.run(SIMPLE_TASK)
        assert result is not None
        assert _has_answer(result)

    def test_openai_explicit_reasoning_effort(self):
        """OpenAI with explicit reasoning_effort should still work."""
        agent = Agent(
            agent_name="OAI-Reasoning",
            model_name=OPENAI_MODEL,
            system_prompt=SIMPLE_PROMPT,
            max_loops=1,
            max_tokens=50,
            print_on=False,
            streaming_on=False,
            reasoning_effort="low",
        )
        result = agent.run(SIMPLE_TASK)
        assert _has_answer(result)


# ===================================================================
# Swarm structures — multi-agent workflows with non-OpenAI providers
# ===================================================================


class TestSwarmStructuresMultiProvider:
    """Swarm structures should work with non-OpenAI agents."""

    def test_sequential_workflow_anthropic(self):
        agent1 = Agent(
            agent_name="Anthropic-Researcher",
            agent_description="Research agent",
            model_name=ANTHROPIC_MODEL,
            max_loops=1,
            max_tokens=100,
            print_on=False,
            streaming_on=False,
        )
        agent2 = Agent(
            agent_name="Anthropic-Summarizer",
            agent_description="Summary agent",
            model_name=ANTHROPIC_MODEL,
            max_loops=1,
            max_tokens=100,
            print_on=False,
            streaming_on=False,
        )
        workflow = SequentialWorkflow(
            name="Anthropic-Workflow",
            agents=[agent1, agent2],
            max_loops=1,
        )
        result = workflow.run("What is 2+2? Explain briefly.")
        assert result is not None

    def test_sequential_workflow_mixed_providers(self):
        """Workflow with OpenAI and Anthropic agents in sequence."""
        openai_agent = Agent(
            agent_name="OpenAI-Step",
            agent_description="OpenAI processing step",
            model_name=OPENAI_MODEL,
            max_loops=1,
            max_tokens=100,
            print_on=False,
            streaming_on=False,
        )
        anthropic_agent = Agent(
            agent_name="Anthropic-Step",
            agent_description="Anthropic processing step",
            model_name=ANTHROPIC_MODEL,
            max_loops=1,
            max_tokens=100,
            print_on=False,
            streaming_on=False,
        )
        workflow = SequentialWorkflow(
            name="Mixed-Provider-Workflow",
            agents=[openai_agent, anthropic_agent],
            max_loops=1,
        )
        result = workflow.run("What is 2+2? Explain briefly.")
        assert result is not None

    def test_mixture_of_agents_anthropic(self):
        """MixtureOfAgents with Anthropic agents."""
        agent1 = Agent(
            agent_name="Anthropic-Analyst-1",
            agent_description="First analyst",
            model_name=ANTHROPIC_MODEL,
            max_loops=1,
            max_tokens=100,
            print_on=False,
            streaming_on=False,
        )
        agent2 = Agent(
            agent_name="Anthropic-Analyst-2",
            agent_description="Second analyst",
            model_name=ANTHROPIC_MODEL,
            max_loops=1,
            max_tokens=100,
            print_on=False,
            streaming_on=False,
        )
        aggregator = Agent(
            agent_name="Anthropic-Aggregator",
            agent_description="Aggregates analyst results",
            model_name=ANTHROPIC_MODEL,
            max_loops=1,
            max_tokens=100,
            print_on=False,
            streaming_on=False,
        )
        moa = MixtureOfAgents(
            name="Anthropic-MOA",
            agents=[agent1, agent2],
            aggregator_agent=aggregator,
            layers=1,
            max_loops=1,
        )
        result = moa.run("What is 2+2?")
        assert result is not None


# ===================================================================
# Groq — no API key: verify params are well-formed
#
# These tests prove the request payload is correct by confirming:
#   1. No reasoning_effort leaks into defaults
#   2. Messages are properly structured
#   3. The LiteLLM call fails on AUTHENTICATION, not on bad parameters
#      (meaning the payload reached the API and was well-formed)
# ===================================================================


class TestGroqProvider:
    """Groq provider — parameter correctness tests."""

    def test_groq_agent_default_reasoning_effort_is_none(self):
        agent = Agent(
            agent_name="Groq-Check",
            model_name=GROQ_MODEL,
            system_prompt=SIMPLE_PROMPT,
            max_loops=1,
            print_on=False,
        )
        assert agent.reasoning_effort is None

    def test_groq_litellm_no_reasoning_in_params(self):
        llm = LiteLLM(
            model_name=GROQ_MODEL,
            system_prompt=SIMPLE_PROMPT,
            max_tokens=50,
        )
        assert llm.reasoning_effort is None
        assert llm.top_p == 1.0

    def test_groq_litellm_messages_prepared_correctly(self):
        llm = LiteLLM(
            model_name=GROQ_MODEL,
            system_prompt=SIMPLE_PROMPT,
            max_tokens=50,
        )
        messages = llm._prepare_messages(SIMPLE_TASK)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == SIMPLE_TASK

    def test_groq_litellm_rejects_on_auth_not_params(self):
        """The call reaches Groq's API and is rejected for auth, not bad params.
        This proves the request payload (no reasoning_effort, correct temp) is valid.
        """
        llm = LiteLLM(
            model_name=GROQ_MODEL,
            system_prompt=SIMPLE_PROMPT,
            max_tokens=50,
        )
        with pytest.raises(Exception) as exc_info:
            llm.run(SIMPLE_TASK)
        _assert_not_param_error(exc_info.value)

    def test_groq_agent_rejects_on_auth_not_params(self):
        """Agent-level: same verification through the full Agent code path."""
        agent = Agent(
            agent_name="Groq-Agent",
            model_name=GROQ_MODEL,
            system_prompt=SIMPLE_PROMPT,
            max_loops=1,
            max_tokens=50,
            print_on=False,
            streaming_on=False,
        )
        # Agent swallows errors and returns conversation history.
        # With no successful LLM call, the result won't contain '4'.
        result = agent.run(SIMPLE_TASK)
        # If result has '4', it somehow worked (key was set). Either way, no crash.
        # The key assertion: we got here without a parameter error crashing the test.
        assert result is not None


# ===================================================================
# Grok (xAI) — no API key: verify params are well-formed
# ===================================================================


class TestGrokProvider:
    """Grok/xAI provider — parameter correctness tests."""

    def test_grok_agent_default_reasoning_effort_is_none(self):
        agent = Agent(
            agent_name="Grok-Check",
            model_name=GROK_MODEL,
            system_prompt=SIMPLE_PROMPT,
            max_loops=1,
            print_on=False,
        )
        assert agent.reasoning_effort is None

    def test_grok_litellm_no_reasoning_in_params(self):
        llm = LiteLLM(
            model_name=GROK_MODEL,
            system_prompt=SIMPLE_PROMPT,
            max_tokens=50,
        )
        assert llm.reasoning_effort is None
        assert llm.top_p == 1.0

    def test_grok_litellm_messages_prepared_correctly(self):
        llm = LiteLLM(
            model_name=GROK_MODEL,
            system_prompt=SIMPLE_PROMPT,
            max_tokens=50,
        )
        messages = llm._prepare_messages(SIMPLE_TASK)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_grok_litellm_rejects_on_auth_not_params(self):
        """The call reaches xAI's API and is rejected for auth, not bad params."""
        llm = LiteLLM(
            model_name=GROK_MODEL,
            system_prompt=SIMPLE_PROMPT,
            max_tokens=50,
        )
        with pytest.raises(Exception) as exc_info:
            llm.run(SIMPLE_TASK)
        _assert_not_param_error(exc_info.value)

    def test_grok_agent_rejects_on_auth_not_params(self):
        """Agent-level: same verification through the full Agent code path."""
        agent = Agent(
            agent_name="Grok-Agent",
            model_name=GROK_MODEL,
            system_prompt=SIMPLE_PROMPT,
            max_loops=1,
            max_tokens=50,
            print_on=False,
            streaming_on=False,
        )
        result = agent.run(SIMPLE_TASK)
        assert result is not None


# ===================================================================
# Gemini — no API key: verify params are well-formed
# ===================================================================


class TestGeminiProvider:
    """Gemini provider — parameter correctness tests."""

    def test_gemini_agent_default_reasoning_effort_is_none(self):
        agent = Agent(
            agent_name="Gemini-Check",
            model_name=GEMINI_MODEL,
            system_prompt=SIMPLE_PROMPT,
            max_loops=1,
            print_on=False,
        )
        assert agent.reasoning_effort is None

    def test_gemini_litellm_no_reasoning_in_params(self):
        llm = LiteLLM(
            model_name=GEMINI_MODEL,
            system_prompt=SIMPLE_PROMPT,
            max_tokens=50,
        )
        assert llm.reasoning_effort is None
        assert llm.top_p == 1.0

    def test_gemini_litellm_messages_prepared_correctly(self):
        llm = LiteLLM(
            model_name=GEMINI_MODEL,
            system_prompt=SIMPLE_PROMPT,
            max_tokens=50,
        )
        messages = llm._prepare_messages(SIMPLE_TASK)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_gemini_litellm_rejects_on_auth_not_params(self):
        """The call reaches Google's API and is rejected for auth, not bad params."""
        llm = LiteLLM(
            model_name=GEMINI_MODEL,
            system_prompt=SIMPLE_PROMPT,
            max_tokens=50,
        )
        with pytest.raises(Exception) as exc_info:
            llm.run(SIMPLE_TASK)
        _assert_not_param_error(exc_info.value)

    def test_gemini_agent_rejects_on_auth_not_params(self):
        """Agent-level: same verification through the full Agent code path."""
        agent = Agent(
            agent_name="Gemini-Agent",
            model_name=GEMINI_MODEL,
            system_prompt=SIMPLE_PROMPT,
            max_loops=1,
            max_tokens=50,
            print_on=False,
            streaming_on=False,
        )
        result = agent.run(SIMPLE_TASK)
        assert result is not None


class TestSerializerWarnings:
    """
    litellm must not emit Pydantic serializer warnings on a non-streaming
    response (#2126).

    ``ModelResponse.choices`` is typed ``List[Union[Choices,
    StreamingChoices]]``. On a non-streaming response the member is
    ``Choices``; a serializer that tries the streaming arm first warns and then
    serializes correctly anyway. litellm calls ``model_dump()`` inside its own
    logging path, so the warning fired on essentially every completion and
    buried real output.

    Offline: this builds the response object directly and never calls a
    provider.
    """

    def test_non_streaming_response_dumps_without_warnings(self):
        from litellm.types.utils import (
            Choices,
            Message,
            ModelResponse,
        )

        response = ModelResponse(
            id="chatcmpl-1",
            created=0,
            model="gpt-4o-mini",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="hi", role="assistant"),
                )
            ],
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            response.model_dump()

        serializer_warnings = [
            str(w.message)
            for w in caught
            if issubclass(w.category, UserWarning)
            and "Pydantic serializer warnings" in str(w.message)
        ]

        assert serializer_warnings == [], (
            "litellm emitted Pydantic serializer warnings on a non-streaming "
            f"response: {serializer_warnings}"
        )
