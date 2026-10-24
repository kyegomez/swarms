"""Regression test: a provider error containing braces must surface as
AgentLLMError (triggering fallback models), not KeyError from loguru
formatting the error text as a template.
"""
from unittest.mock import patch

import pytest

from swarms import Agent
from swarms.schemas.agent_errors import AgentLLMError
from swarms.utils import litellm_wrapper

BRACED_ERROR = "Error code: 404 - [{'error': {'code': 404}}]"


def test_run_error_with_braces_raises_agent_llm_error():
    agent = Agent(
        agent_name="A",
        model_name="gpt-5.4",
        max_loops=1,
        print_on=False,
        persistent_memory=False,
    )
    with patch.object(
        litellm_wrapper, "completion", side_effect=RuntimeError(BRACED_ERROR)
    ):
        with pytest.raises(AgentLLMError):
            agent.run("hi")


def test_run_error_with_braces_does_not_raise_key_error():
    agent = Agent(
        agent_name="A",
        model_name="gpt-5.4",
        max_loops=1,
        print_on=False,
        persistent_memory=False,
    )
    with patch.object(
        litellm_wrapper, "completion", side_effect=RuntimeError(BRACED_ERROR)
    ):
        try:
            agent.run("hi")
        except KeyError as exc:
            pytest.fail(f"loguru formatted the error text as a template: {exc}")
        except AgentLLMError:
            pass
