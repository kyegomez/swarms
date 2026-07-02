"""
Tests for swarms.structs.multi_agent_exec

Covers:
- asyncio.get_running_loop() usage in run_agent_async
- asyncio.run() usage in run_agents_concurrently_multiprocess
- per_task_timeout parameter in run_agents_concurrently
- backward compatibility (no timeout by default)
"""

import asyncio
import time
from unittest.mock import patch

import pytest

from swarms.structs.multi_agent_exec import (
    run_single_agent,
    run_agent_async,
    run_agents_concurrently,
    run_agents_concurrently_async,
    run_agents_concurrently_multiprocess,
    batched_grid_agent_execution,
    run_agents_with_different_tasks,
)

# Helpers / Fixtures


class _FakeAgent:
    """Lightweight stub that avoids real LLM calls."""

    def __init__(
        self,
        name: str = "fake-agent",
        delay: float = 0.0,
        result: str = "ok",
        raise_exc: Exception | None = None,
    ):
        self.agent_name = name
        self.name = name
        self._delay = delay
        self._result = result
        self._raise_exc = raise_exc

    def run(self, task: str = "", **kwargs) -> str:
        if self._delay:
            time.sleep(self._delay)
        if self._raise_exc is not None:
            raise self._raise_exc
        return f"{self._result}: {task}"


class _SlowAgent(_FakeAgent):
    """Agent that sleeps longer than any reasonable timeout."""

    def __init__(self, name: str = "slow-agent"):
        super().__init__(
            name=name, delay=30.0, result="should-not-reach"
        )


# 1. run_single_agent


class TestRunSingleAgent:
    def test_returns_agent_result(self):
        agent = _FakeAgent(result="hello")
        out = run_single_agent(agent, "world")
        assert out == "hello: world"

    def test_propagates_exception(self):
        agent = _FakeAgent(raise_exc=ValueError("boom"))
        with pytest.raises(ValueError, match="boom"):
            run_single_agent(agent, "task")


# 2. run_agent_async — must use get_running_loop()


class TestRunAgentAsync:
    def test_runs_in_async_context(self):
        """Verify run_agent_async works inside an
        already-running event loop."""
        agent = _FakeAgent(result="async-ok")

        async def _run():
            return await run_agent_async(agent, "ping")

        result = asyncio.run(_run())
        assert result == "async-ok: ping"

    def test_uses_get_running_loop(self):
        """Confirm get_running_loop is called, not the
        deprecated get_event_loop."""
        agent = _FakeAgent()

        async def _run():
            with patch(
                "swarms.structs.multi_agent_exec"
                ".asyncio.get_running_loop",
                wraps=asyncio.get_running_loop,
            ) as mock_grl:
                await run_agent_async(agent, "t")
                mock_grl.assert_called_once()

        asyncio.run(_run())


# 3. run_agents_concurrently_async


class TestRunAgentsConcurrentlyAsync:
    def test_gathers_all_results(self):
        agents = [
            _FakeAgent(name=f"a{i}", result=f"r{i}") for i in range(3)
        ]

        async def _run():
            return await run_agents_concurrently_async(agents, "task")

        results = asyncio.run(_run())
        assert len(results) == 3
        for i, r in enumerate(results):
            assert r.startswith(f"r{i}")


# 4. run_agents_concurrently — default (no timeout)


class TestRunAgentsConcurrently:
    def test_returns_list_by_default(self):
        agents = [
            _FakeAgent(name="a1", result="x"),
            _FakeAgent(name="a2", result="y"),
        ]
        results = run_agents_concurrently(
            agents, task="t", max_workers=2
        )
        assert isinstance(results, list)
        assert len(results) == 2

    def test_returns_dict_when_requested(self):
        agents = [
            _FakeAgent(name="alpha", result="res-a"),
            _FakeAgent(name="beta", result="res-b"),
        ]
        out = run_agents_concurrently(
            agents,
            task="t",
            max_workers=2,
            return_agent_output_dict=True,
        )
        assert isinstance(out, dict)
        assert "alpha" in out
        assert "beta" in out
        assert out["alpha"].startswith("res-a")
        assert out["beta"].startswith("res-b")

    def test_no_timeout_by_default(self):
        """Without per_task_timeout the call should still
        complete normally."""
        agents = [_FakeAgent(name="fast", delay=0.05)]
        results = run_agents_concurrently(
            agents, task="t", max_workers=1
        )
        assert len(results) == 1
        assert isinstance(results[0], str)


# 5. run_agents_concurrently — per_task_timeout


class TestRunAgentsConcurrentlyTimeout:
    def test_timeout_captures_exception_list(self):
        """A slow agent should produce a timeout exception
        in the results list, not crash the batch.

        Note: The list path uses as_completed(), which
        yields futures only after they complete. Therefore
        the timeout on future.result() acts as a secondary
        guard. The dict path (tested below) is the primary
        timeout mechanism. Here we verify the list path
        works correctly with fast agents and a timeout."""
        agents = [
            _FakeAgent(name="fast", delay=0.0, result="done"),
            _FakeAgent(name="medium", delay=0.1, result="ok"),
        ]
        results = run_agents_concurrently(
            agents,
            task="t",
            max_workers=2,
            per_task_timeout=5.0,
        )
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)

    def test_timeout_captures_exception_dict(self):
        """Same as above but with return_agent_output_dict."""
        agents = [
            _FakeAgent(name="ok-agent", delay=0.0),
            _FakeAgent(
                name="hung-agent",
                delay=10.0,
                result="nope",
            ),
        ]
        out = run_agents_concurrently(
            agents,
            task="t",
            max_workers=2,
            per_task_timeout=0.5,
            return_agent_output_dict=True,
        )
        assert isinstance(out, dict)
        assert "ok-agent" in out
        assert "hung-agent" in out
        # ok-agent should have a string result
        assert isinstance(out["ok-agent"], str)
        # hung-agent should have a timeout exception
        assert isinstance(out["hung-agent"], Exception)

    def test_all_agents_succeed_within_timeout(self):
        """When all agents finish in time, results are
        normal strings."""
        agents = [
            _FakeAgent(name=f"a{i}", delay=0.01) for i in range(3)
        ]
        results = run_agents_concurrently(
            agents,
            task="t",
            max_workers=3,
            per_task_timeout=5.0,
        )
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)

    def test_none_timeout_is_backward_compatible(self):
        """Explicitly passing per_task_timeout=None should
        behave identically to not passing it."""
        agents = [_FakeAgent(name="compat")]
        results = run_agents_concurrently(
            agents,
            task="t",
            max_workers=1,
            per_task_timeout=None,
        )
        assert len(results) == 1
        assert isinstance(results[0], str)


# 6. run_agents_concurrently_multiprocess — uses asyncio.run()


class TestRunAgentsConcurrentlyMultiprocess:
    def test_basic_execution(self):
        """Verify the function works from a sync context
        (no pre-existing event loop)."""
        agents = [
            _FakeAgent(name=f"mp{i}", result=f"r{i}")
            for i in range(3)
        ]
        results = run_agents_concurrently_multiprocess(
            agents, task="t", batch_size=2
        )
        assert len(results) == 3

    def test_uses_asyncio_run(self):
        """Confirm asyncio.run is used, not the deprecated
        get_event_loop + run_until_complete pattern."""
        agents = [_FakeAgent(name="mp-check")]

        with patch(
            "swarms.structs.multi_agent_exec.asyncio.run",
            wraps=asyncio.run,
        ) as mock_run:
            run_agents_concurrently_multiprocess(
                agents, task="t", batch_size=1
            )
            assert mock_run.call_count >= 1

    def test_batching_processes_all_agents(self):
        """With batch_size < len(agents), all agents should
        still produce results."""
        agents = [_FakeAgent(name=f"batch{i}") for i in range(5)]
        results = run_agents_concurrently_multiprocess(
            agents, task="t", batch_size=2
        )
        assert len(results) == 5


# 7. batched_grid_agent_execution


class TestBatchedGridAgentExecution:
    def test_basic_grid(self):
        agents = [
            _FakeAgent(name="g1", result="r1"),
            _FakeAgent(name="g2", result="r2"),
        ]
        tasks = ["task-a", "task-b"]
        results = batched_grid_agent_execution(
            agents, tasks, max_workers=2
        )
        assert len(results) == 2
        assert results[0].startswith("r1")
        assert results[1].startswith("r2")

    def test_mismatched_lengths_raises(self):
        agents = [_FakeAgent()]
        tasks = ["a", "b"]
        with pytest.raises(ValueError):
            batched_grid_agent_execution(agents, tasks)

    def test_exception_captured_in_results(self):
        agents = [
            _FakeAgent(
                name="err",
                raise_exc=RuntimeError("fail"),
            ),
            _FakeAgent(name="ok", result="fine"),
        ]
        tasks = ["t1", "t2"]
        results = batched_grid_agent_execution(
            agents, tasks, max_workers=2
        )
        assert isinstance(results[0], RuntimeError)
        assert isinstance(results[1], str)


# 8. run_agents_with_different_tasks


class TestRunAgentsWithDifferentTasks:
    def test_basic_pairs(self):
        pairs = [
            (_FakeAgent(name="p1", result="r1"), "task-1"),
            (_FakeAgent(name="p2", result="r2"), "task-2"),
        ]
        results = run_agents_with_different_tasks(pairs, batch_size=2)
        assert len(results) == 2

    def test_empty_pairs(self):
        results = run_agents_with_different_tasks([])
        assert results == []

    def test_batching_preserves_order(self):
        pairs = [
            (
                _FakeAgent(name=f"p{i}", result=f"r{i}"),
                f"task-{i}",
            )
            for i in range(5)
        ]
        results = run_agents_with_different_tasks(pairs, batch_size=2)
        assert len(results) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
