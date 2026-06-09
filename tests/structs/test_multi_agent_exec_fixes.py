"""Tests for multi_agent_exec fixes: deprecated event loop and per-task timeout."""

import time
from unittest.mock import MagicMock

from swarms.structs.multi_agent_exec import (
    run_agents_concurrently,
    run_agents_concurrently_multiprocess,
)


def _agent(name: str, output: str = "ok", delay: float = 0.0):
    a = MagicMock()
    a.agent_name = name

    def slow_run(**kwargs):
        if delay:
            time.sleep(delay)
        return output

    a.run = slow_run
    return a


# ---------------------------------------------------------------------------
# asyncio.run() replacement for run_agents_concurrently_multiprocess
# ---------------------------------------------------------------------------


def test_multiprocess_runner_returns_results():
    agents = [_agent("A", "a"), _agent("B", "b")]
    results = run_agents_concurrently_multiprocess(
        agents, task="t", batch_size=2
    )
    assert set(results) == {"a", "b"}


def test_multiprocess_runner_batches_correctly():
    agents = [_agent(str(i), str(i)) for i in range(4)]
    results = run_agents_concurrently_multiprocess(
        agents, task="t", batch_size=2
    )
    assert len(results) == 4


# ---------------------------------------------------------------------------
# per_task_timeout — dict return path
# ---------------------------------------------------------------------------


def test_timeout_dict_returns_timeout_error_for_slow_agent():
    agents = [
        _agent("Fast", "done"),
        _agent("Slow", "never", delay=0.5),
    ]
    result = run_agents_concurrently(
        agents,
        task="t",
        return_agent_output_dict=True,
        per_task_timeout=0.1,
    )
    assert result["Fast"] == "done"
    assert isinstance(result["Slow"], TimeoutError)


def test_no_timeout_dict_completes_normally():
    agents = [_agent("A", "a"), _agent("B", "b")]
    result = run_agents_concurrently(
        agents,
        task="t",
        return_agent_output_dict=True,
        per_task_timeout=None,
    )
    assert result == {"A": "a", "B": "b"}


# ---------------------------------------------------------------------------
# per_task_timeout — list return path
# ---------------------------------------------------------------------------


def test_timeout_list_returns_timeout_error_for_slow_agent():
    agents = [
        _agent("Fast", "done"),
        _agent("Slow", "never", delay=0.5),
    ]
    results = run_agents_concurrently(
        agents,
        task="t",
        return_agent_output_dict=False,
        per_task_timeout=0.1,
    )
    assert any(r == "done" for r in results)
    assert any(isinstance(r, TimeoutError) for r in results)


def test_no_timeout_list_completes_normally():
    agents = [_agent("A", "a"), _agent("B", "b")]
    results = run_agents_concurrently(
        agents,
        task="t",
        return_agent_output_dict=False,
        per_task_timeout=None,
    )
    assert set(results) == {"a", "b"}
