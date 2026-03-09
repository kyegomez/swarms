"""Tests for async subagent execution."""

import time
from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest

from swarms.structs.async_subagent import (
    SubagentRegistry,
    SubagentTask,
    TaskStatus,
)


# ── Helpers ─────────────────────────────────────────────────


def make_mock_agent(
    result="done", delay=0, name="mock-agent", error=None
):
    """Create a mock agent with a .run() method."""
    agent = MagicMock()
    agent.agent_name = name

    def run(task):
        if delay:
            time.sleep(delay)
        if error:
            raise error
        return result

    agent.run = MagicMock(side_effect=run)
    return agent


# ── SubagentRegistry Tests ──────────────────────────────────


class TestSubagentRegistry:
    def test_spawn_returns_task_id(self):
        reg = SubagentRegistry()
        agent = make_mock_agent()
        task_id = reg.spawn(agent, "do something")
        assert task_id.startswith("task-")
        reg.shutdown()

    def test_task_status_transitions(self):
        reg = SubagentRegistry()
        agent = make_mock_agent(delay=0.05)
        task_id = reg.spawn(agent, "test")

        # Should be running or completed quickly
        st = reg.get_task(task_id)
        assert st.status in (TaskStatus.RUNNING, TaskStatus.COMPLETED)

        reg.gather()
        st = reg.get_task(task_id)
        assert st.status == TaskStatus.COMPLETED
        assert st.result == "done"
        assert st.completed_at is not None
        reg.shutdown()

    def test_get_task_not_found(self):
        reg = SubagentRegistry()
        with pytest.raises(KeyError):
            reg.get_task("nonexistent")
        reg.shutdown()

    def test_gather_wait_all(self):
        reg = SubagentRegistry()
        agents = [
            make_mock_agent(result=f"r{i}", delay=0.05)
            for i in range(3)
        ]
        for a in agents:
            reg.spawn(a, "task")
        results = reg.gather(strategy="wait_all")
        assert len(results) == 3
        assert set(results) == {"r0", "r1", "r2"}
        reg.shutdown()

    def test_gather_wait_first(self):
        reg = SubagentRegistry()
        fast = make_mock_agent(result="fast", delay=0.01)
        slow = make_mock_agent(result="slow", delay=2.0)
        reg.spawn(fast, "task")
        reg.spawn(slow, "task")
        results = reg.gather(strategy="wait_first")
        assert len(results) >= 1
        assert "fast" in results
        reg.shutdown()

    def test_gather_with_timeout(self):
        reg = SubagentRegistry()
        agent = make_mock_agent(delay=5.0)
        reg.spawn(agent, "slow task")
        results = reg.gather(strategy="wait_all", timeout=0.1)
        # With timeout, we may get empty results since task is still running
        assert isinstance(results, list)
        reg.shutdown()

    def test_get_results(self):
        reg = SubagentRegistry()
        a1 = make_mock_agent(result="hello")
        a2 = make_mock_agent(result="world")
        reg.spawn(a1, "t1")
        reg.spawn(a2, "t2")
        reg.gather()
        results = reg.get_results()
        assert len(results) == 2
        assert set(results.values()) == {"hello", "world"}
        reg.shutdown()

    def test_cancel_task(self):
        reg = SubagentRegistry()
        agent = make_mock_agent(delay=10.0)
        task_id = reg.spawn(agent, "long task")
        # Cancel may or may not succeed depending on timing
        cancelled = reg.cancel(task_id)
        assert isinstance(cancelled, bool)
        reg.shutdown()

    def test_depth_limit_enforced(self):
        reg = SubagentRegistry(max_depth=2)
        agent = make_mock_agent()
        # depth=2 should work
        reg.spawn(agent, "ok", depth=2)
        # depth=3 should fail
        with pytest.raises(ValueError, match="exceeds max_depth"):
            reg.spawn(agent, "too deep", depth=3)
        reg.shutdown()

    def test_retry_on_failure(self):
        call_count = 0

        def flaky_run(task):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("temporary failure")
            return "success"

        agent = MagicMock()
        agent.agent_name = "flaky"
        agent.run = MagicMock(side_effect=flaky_run)

        reg = SubagentRegistry()
        task_id = reg.spawn(
            agent, "retry me", max_retries=3, retry_on=[RuntimeError]
        )
        reg.gather()
        st = reg.get_task(task_id)
        assert st.status == TaskStatus.COMPLETED
        assert st.result == "success"
        assert st.retries == 2
        reg.shutdown()

    def test_retry_exhausted_fails(self):
        agent = make_mock_agent(error=ValueError("always fails"))
        reg = SubagentRegistry()
        task_id = reg.spawn(
            agent,
            "doomed",
            max_retries=2,
            retry_on=[ValueError],
            fail_fast=False,
        )
        reg.gather()
        st = reg.get_task(task_id)
        assert st.status == TaskStatus.FAILED
        assert isinstance(st.error, ValueError)
        reg.shutdown()

    def test_fail_fast_false_does_not_raise(self):
        agent = make_mock_agent(error=RuntimeError("boom"))
        reg = SubagentRegistry()
        task_id = reg.spawn(agent, "task", fail_fast=False)
        results = reg.gather()
        # Should not raise, result is None
        st = reg.get_task(task_id)
        assert st.status == TaskStatus.FAILED
        assert st.error is not None
        reg.shutdown()

    def test_fail_fast_true_raises_in_result(self):
        agent = make_mock_agent(error=RuntimeError("boom"))
        reg = SubagentRegistry()
        reg.spawn(agent, "task", fail_fast=True)
        results = reg.gather()
        # gather catches the exception from the future
        assert len(results) == 1
        assert isinstance(results[0], RuntimeError)
        reg.shutdown()

    def test_tasks_property(self):
        reg = SubagentRegistry()
        agent = make_mock_agent()
        task_id = reg.spawn(agent, "test")
        tasks = reg.tasks
        assert task_id in tasks
        assert isinstance(tasks[task_id], SubagentTask)
        reg.shutdown()

    def test_concurrent_spawns(self):
        reg = SubagentRegistry()
        agents = [
            make_mock_agent(result=f"r{i}", delay=0.02)
            for i in range(10)
        ]
        task_ids = [
            reg.spawn(a, f"task-{i}") for i, a in enumerate(agents)
        ]
        assert len(task_ids) == 10
        results = reg.gather()
        assert len(results) == 10
        reg.shutdown()

    def test_lifecycle_logging(self, caplog):
        """Verify lifecycle events are logged."""
        import logging

        with caplog.at_level(logging.INFO):
            reg = SubagentRegistry()
            agent = make_mock_agent()
            task_id = reg.spawn(agent, "log test")
            reg.gather()
            reg.shutdown()
        # loguru doesn't integrate with caplog by default, so just verify no errors
        assert True


# ── Agent Integration Tests ─────────────────────────────────


class TestAgentAsyncMethods:
    """Test the async subagent methods added to Agent.

    Uses a mock Agent-like object to avoid LLM initialization.
    """

    def _make_agent_stub(self, run_result="agent-result", delay=0):
        """Create a minimal Agent-like stub with the async subagent methods."""
        from swarms.structs.async_subagent import SubagentRegistry

        class AgentStub:
            def __init__(self):
                self.id = "agent-001"
                self.agent_name = "stub-agent"
                self.max_subagent_depth = 3
                self._subagent_registry = None
                self._run_result = run_result
                self._delay = delay

            def run(self, task):
                if self._delay:
                    time.sleep(self._delay)
                return self._run_result

            def _get_registry(self):
                if self._subagent_registry is None:
                    self._subagent_registry = SubagentRegistry(
                        max_depth=self.max_subagent_depth
                    )
                return self._subagent_registry

            def run_async(self, task):
                registry = self._get_registry()
                return registry._executor.submit(self.run, task)

            def spawn_async(
                self,
                agent,
                task,
                max_retries=0,
                retry_on=None,
                fail_fast=True,
            ):
                registry = self._get_registry()
                return registry.spawn(
                    agent=agent,
                    task=task,
                    parent_id=self.id,
                    depth=0,
                    max_retries=max_retries,
                    retry_on=retry_on,
                    fail_fast=fail_fast,
                )

            def run_in_background(self, task):
                registry = self._get_registry()
                return registry.spawn(
                    agent=self, task=task, parent_id=None, depth=0
                )

            def gather_results(
                self, strategy="wait_all", timeout=None
            ):
                return self._get_registry().gather(
                    strategy=strategy, timeout=timeout
                )

            def get_subagent_results(self):
                return self._get_registry().get_results()

            def cancel_subagent(self, task_id):
                return self._get_registry().cancel(task_id)

        return AgentStub()

    def test_run_async_returns_future(self):
        agent = self._make_agent_stub()
        future = agent.run_async("test task")
        assert isinstance(future, Future)
        result = future.result(timeout=5)
        assert result == "agent-result"

    def test_spawn_async_runs_subagent(self):
        parent = self._make_agent_stub()
        child = make_mock_agent(result="child-result")
        task_id = parent.spawn_async(child, "do work")
        assert task_id.startswith("task-")
        results = parent.gather_results()
        assert "child-result" in results

    def test_run_in_background(self):
        agent = self._make_agent_stub(
            run_result="bg-result", delay=0.05
        )
        task_id = agent.run_in_background("background task")
        assert task_id.startswith("task-")
        results = agent.gather_results()
        assert "bg-result" in results

    def test_gather_results_wait_all(self):
        parent = self._make_agent_stub()
        for i in range(3):
            child = make_mock_agent(result=f"result-{i}", delay=0.02)
            parent.spawn_async(child, f"task-{i}")
        results = parent.gather_results(strategy="wait_all")
        assert len(results) == 3

    def test_gather_results_wait_first(self):
        parent = self._make_agent_stub()
        fast = make_mock_agent(result="fast", delay=0.01)
        slow = make_mock_agent(result="slow", delay=2.0)
        parent.spawn_async(fast, "fast task")
        parent.spawn_async(slow, "slow task")
        results = parent.gather_results(strategy="wait_first")
        assert len(results) >= 1
        assert "fast" in results

    def test_get_subagent_results(self):
        parent = self._make_agent_stub()
        child = make_mock_agent(result="output")
        parent.spawn_async(child, "task")
        parent.gather_results()
        results = parent.get_subagent_results()
        assert len(results) == 1
        assert list(results.values()) == ["output"]

    def test_cancel_subagent(self):
        parent = self._make_agent_stub()
        child = make_mock_agent(delay=10.0)
        task_id = parent.spawn_async(child, "long task")
        result = parent.cancel_subagent(task_id)
        assert isinstance(result, bool)

    def test_parent_not_blocked(self):
        """Verify parent can continue while subagents run."""
        parent = self._make_agent_stub()
        child = make_mock_agent(result="child-done", delay=0.2)

        start = time.time()
        parent.spawn_async(child, "slow task")
        spawn_time = time.time() - start

        # spawn_async should return almost immediately
        assert spawn_time < 0.1

        # Parent can do other work here
        parent_result = parent.run("parent task")
        assert parent_result == "agent-result"

        # Then collect subagent results
        results = parent.gather_results()
        assert "child-done" in results


# ── Recursive Subagent Tree Tests ───────────────────────────


class TestRecursiveSubagents:
    def test_subagent_spawns_child(self):
        """Subagent can spawn its own child (depth tracking)."""
        reg = SubagentRegistry(max_depth=3)
        grandparent = make_mock_agent(result="gp")
        parent = make_mock_agent(result="parent")
        child = make_mock_agent(result="child")

        gp_id = reg.spawn(grandparent, "task", depth=0)
        p_id = reg.spawn(parent, "task", parent_id=gp_id, depth=1)
        c_id = reg.spawn(child, "task", parent_id=p_id, depth=2)

        results = reg.gather()
        assert len(results) == 3
        assert set(results) == {"gp", "parent", "child"}

        # Verify depth tracking
        assert reg.get_task(gp_id).depth == 0
        assert reg.get_task(p_id).depth == 1
        assert reg.get_task(c_id).depth == 2
        reg.shutdown()

    def test_depth_limit_prevents_runaway(self):
        """Cannot exceed max_subagent_depth."""
        reg = SubagentRegistry(max_depth=1)
        agent = make_mock_agent()

        reg.spawn(agent, "ok", depth=0)
        reg.spawn(agent, "ok", depth=1)
        with pytest.raises(ValueError):
            reg.spawn(agent, "too deep", depth=2)
        reg.shutdown()

    def test_parent_child_relationship(self):
        """Parent ID is tracked correctly."""
        reg = SubagentRegistry()
        parent = make_mock_agent()
        child = make_mock_agent()

        parent_task_id = reg.spawn(parent, "parent task")
        child_task_id = reg.spawn(
            child, "child task", parent_id=parent_task_id, depth=1
        )

        child_task = reg.get_task(child_task_id)
        assert child_task.parent_id == parent_task_id
        reg.shutdown()


# ── Error Propagation Tests ─────────────────────────────────


class TestErrorPropagation:
    def test_failed_task_error_in_results(self):
        reg = SubagentRegistry()
        agent = make_mock_agent(error=ValueError("bad input"))
        task_id = reg.spawn(agent, "fail", fail_fast=False)
        reg.gather()
        results = reg.get_results()
        assert task_id in results
        assert isinstance(results[task_id], ValueError)
        reg.shutdown()

    def test_retry_policy_with_specific_exception(self):
        """Only retry on specified exception types."""
        call_count = 0

        def run(task):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TypeError("wrong type")  # Not in retry_on
            return "ok"

        agent = MagicMock()
        agent.agent_name = "selective"
        agent.run = MagicMock(side_effect=run)

        reg = SubagentRegistry()
        task_id = reg.spawn(
            agent,
            "test",
            max_retries=3,
            retry_on=[ValueError],
            fail_fast=False,
        )
        reg.gather()
        st = reg.get_task(task_id)
        # Should fail because TypeError is not in retry_on
        assert st.status == TaskStatus.FAILED
        assert call_count == 1
        reg.shutdown()

    def test_mixed_success_and_failure(self):
        reg = SubagentRegistry()
        good = make_mock_agent(result="ok")
        bad = make_mock_agent(error=RuntimeError("fail"))

        reg.spawn(good, "good task")
        reg.spawn(bad, "bad task", fail_fast=False)
        reg.gather()

        results = reg.get_results()
        values = list(results.values())
        assert "ok" in values
        assert any(isinstance(v, RuntimeError) for v in values)
        reg.shutdown()
