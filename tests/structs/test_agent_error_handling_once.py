import asyncio
from unittest.mock import MagicMock
import pytest

from swarms.structs.agent import Agent


def test_agent_call_handles_error_only_once():
    """Agent.__call__ must not catch and re-handle errors already handled by run()."""
    agent = Agent(
        agent_name="TestAgent",
        model_name="gpt-5.4",
        max_loops=1,
        autosave=False,
        print_on=False,
    )

    handle_calls = []
    orig_handle = agent._handle_run_error

    def spy_handle(err):
        handle_calls.append(err)
        return orig_handle(err)

    agent._handle_run_error = spy_handle

    # Mock _run so it invokes _handle_run_error and re-raises, exactly as real _run does
    def failing_run(*args, **kwargs):
        err = RuntimeError("simulated agent error")
        agent._handle_run_error(err)
        raise err

    agent._run = failing_run

    with pytest.raises(RuntimeError, match="simulated agent error"):
        agent("failing task")

    assert len(handle_calls) == 1, f"Expected 1 call to _handle_run_error, got {len(handle_calls)}"


def test_agent_arun_handles_error_only_once():
    """Agent.arun must not catch and re-handle errors already handled by run()."""
    agent = Agent(
        agent_name="TestAgentAsync",
        model_name="gpt-5.4",
        max_loops=1,
        autosave=False,
        print_on=False,
    )

    handle_calls = []
    orig_handle = agent._handle_run_error

    def spy_handle(err):
        handle_calls.append(err)
        return orig_handle(err)

    agent._handle_run_error = spy_handle

    def failing_run(*args, **kwargs):
        err = RuntimeError("simulated async error")
        agent._handle_run_error(err)
        raise err

    agent._run = failing_run

    with pytest.raises(RuntimeError, match="simulated async error"):
        asyncio.run(agent.arun("failing async task"))

    assert len(handle_calls) == 1, f"Expected 1 call to _handle_run_error, got {len(handle_calls)}"
