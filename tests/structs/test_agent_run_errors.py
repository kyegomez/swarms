"""Tests for Agent failure honesty.

Covers the correctness fix: when the LLM fails on every retry attempt,
``Agent.run`` must raise ``AgentLLMError`` instead of silently returning
the raw conversation transcript as if it were the answer.
"""

import pytest
from litellm.exceptions import BadRequestError

from swarms.schemas.agent_errors import AgentError
from swarms.structs.agent import (
    Agent,
    AgentLLMError,
)


class _FailingAgent(Agent):
    """Agent whose LLM call always raises a retryable litellm error."""

    def call_llm(self, *args, **kwargs):
        raise BadRequestError(
            message="stub failure: model unreachable",
            model="gpt-5.4",
            llm_provider="stub",
        )


class TestFailureHonesty:
    def test_run_raises_agent_llm_error_after_retries(self):
        agent = _FailingAgent(
            agent_name="fail-honest",
            retry_attempts=2,
            max_loops=1,
            persistent_memory=False,
        )
        with pytest.raises(AgentLLMError):
            agent.run("hello")

    def test_error_message_reports_attempts(self):
        agent = _FailingAgent(
            agent_name="fail-honest-msg",
            retry_attempts=3,
            max_loops=1,
            persistent_memory=False,
        )
        with pytest.raises(AgentLLMError) as excinfo:
            agent.run("hello")
        assert "3 retry attempt" in str(excinfo.value)

    def test_agent_llm_error_is_an_agent_error(self):
        assert issubclass(AgentLLMError, AgentError)