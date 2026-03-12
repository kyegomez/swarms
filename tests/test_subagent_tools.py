"""
Comprehensive tests for the subagent tool system in autonomous_loop_utils.py.
All tests use real Agent instances with real LLM calls (gpt-4.1-nano).
No mocks.

Run:
    python -m pytest tests/test_subagent_tools.py -v
    python tests/test_subagent_tools.py
"""

import time

import pytest
from dotenv import load_dotenv

load_dotenv()

from swarms.structs.agent import Agent
from swarms.structs.autonomous_loop_utils import (
    SubagentTaskStatus,
    SubagentTaskRegistry,
    _get_task_registry,
    create_sub_agent_tool,
    assign_task_tool,
    get_task_status_tool,
    cancel_task_tool,
)


MODEL = "gpt-4.1-nano"
PROMPT = "You are a helpful assistant. Be very brief, one sentence max."


def make_real_agent(name="worker"):
    return Agent(
        agent_name=name,
        system_prompt=PROMPT,
        model_name=MODEL,
        max_loops=1,
        print_on=False,
        streaming_on=False,
        verbose=False,
    )


def make_real_parent():
    return Agent(
        agent_name="parent",
        system_prompt="You are a coordinator agent.",
        model_name=MODEL,
        max_loops=1,
        print_on=False,
        streaming_on=False,
        verbose=False,
    )


# ── Registry Tests (real agents) ────────────────────────────


class TestRegistry:
    def test_spawn_and_gather(self):
        reg = SubagentTaskRegistry()
        agent = make_real_agent("math")
        task_id = reg.spawn(agent, "math-id", "What is 2+2? One number only.")
        assert task_id.startswith("task-")
        results = reg.gather()
        assert len(results) == 1
        assert isinstance(results[0], str) and len(results[0]) > 0
        st = reg.get_task(task_id)
        assert st.status == SubagentTaskStatus.COMPLETED
        assert st.completed_at is not None
        reg.shutdown()

    def test_concurrent_agents(self):
        reg = SubagentTaskRegistry()
        agents = [make_real_agent(f"w{i}") for i in range(3)]
        for i, a in enumerate(agents):
            reg.spawn(a, f"a{i}", f"What is {i}+{i}? One number only.")
        results = reg.gather()
        assert len(results) == 3
        for r in results:
            assert isinstance(r, str) and len(r) > 0
        reg.shutdown()

    def test_gather_wait_first(self):
        reg = SubagentTaskRegistry()
        a1 = make_real_agent("fast")
        a2 = make_real_agent("slow")
        reg.spawn(a1, "a1", "Say 'done'. One word.")
        reg.spawn(a2, "a2", "Write a 500-word essay on quantum physics.")
        start = time.time()
        results = reg.gather(strategy="wait_first")
        elapsed = time.time() - start
        assert len(results) >= 1
        reg.shutdown()

    def test_depth_limit(self):
        reg = SubagentTaskRegistry(max_depth=1)
        agent = make_real_agent()
        reg.spawn(agent, "a1", "hello", depth=1)
        with pytest.raises(ValueError, match="exceeds max_depth"):
            reg.spawn(agent, "a1", "too deep", depth=2)
        reg.shutdown()

    def test_retry_on_failure(self):
        """Use a real agent with broken model to force failures, then succeed."""
        call_count = 0
        good_agent = make_real_agent("flaky")
        original_run = good_agent.run

        def flaky_run(task):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("simulated failure")
            return original_run(task)

        good_agent.run = flaky_run

        reg = SubagentTaskRegistry()
        task_id = reg.spawn(
            good_agent, "flaky-id", "Say 'recovered'. One word.",
            max_retries=3, retry_on=[RuntimeError],
        )
        reg.gather()
        st = reg.get_task(task_id)
        assert st.status == SubagentTaskStatus.COMPLETED
        assert isinstance(st.result, str)
        assert st.retries == 2
        reg.shutdown()

    def test_fail_fast_false(self):
        """Agent that raises in .run() gets FAILED status, doesn't crash registry."""

        class CrashingAgent:
            agent_name = "crasher"
            def run(self, task):
                raise RuntimeError("hard crash")

        reg = SubagentTaskRegistry()
        task_id = reg.spawn(CrashingAgent(), "crash-id", "will fail", fail_fast=False)
        reg.gather()
        st = reg.get_task(task_id)
        assert st.status == SubagentTaskStatus.FAILED
        assert isinstance(st.error, RuntimeError)
        reg.shutdown()


# ── create_sub_agent_tool Tests ──────────────────────────────


class TestCreateSubAgent:
    def test_system_prompt_default_when_none(self):
        """The bug fix: missing system_prompt gets a default, not None."""
        parent = make_real_parent()
        result = create_sub_agent_tool(
            parent,
            [{"agent_name": "Greeter", "agent_description": "Greets users in English"}],
        )
        assert "Successfully created" in result
        data = list(parent.sub_agents.values())[0]
        assert data["system_prompt"] is not None
        assert "Greeter" in data["system_prompt"]
        assert "Greets users" in data["system_prompt"]

    def test_sub_agent_actually_runs(self):
        """Sub-agent created without system_prompt can execute without crash."""
        parent = make_real_parent()
        create_sub_agent_tool(
            parent,
            [{"agent_name": "Helper", "agent_description": "Answers questions briefly"}],
        )
        sub_agent = list(parent.sub_agents.values())[0]["agent"]
        output = sub_agent.run("What is 1+1? One number only.")
        assert isinstance(output, str) and len(output) > 0

    def test_depth_tracking(self):
        parent = make_real_parent()
        parent.max_subagent_depth = 2

        # Create child (depth 1)
        create_sub_agent_tool(
            parent,
            [{"agent_name": "Child", "agent_description": "Child agent"}],
        )
        child = list(parent.sub_agents.values())[0]["agent"]
        assert child._subagent_depth == 1

        # Child creates grandchild (depth 2)
        create_sub_agent_tool(
            child,
            [{"agent_name": "Grandchild", "agent_description": "Grandchild agent"}],
        )
        grandchild = list(child.sub_agents.values())[0]["agent"]
        assert grandchild._subagent_depth == 2

        # Grandchild blocked at depth 3
        result = create_sub_agent_tool(
            grandchild,
            [{"agent_name": "TooDeep", "agent_description": "Blocked"}],
        )
        assert "Error" in result
        assert "Maximum subagent depth" in result

    def test_missing_agent_name(self):
        parent = make_real_parent()
        result = create_sub_agent_tool(
            parent,
            [{"agent_description": "No name provided"}],
        )
        assert "Error" in result


# ── assign_task_tool Tests ───────────────────────────────────


class TestAssignTask:
    def _create_sub_agents(self, parent, specs):
        """Helper: create sub-agents and return their IDs."""
        create_sub_agent_tool(parent, specs)
        return list(parent.sub_agents.keys())

    def test_basic_assign_and_wait(self):
        parent = make_real_parent()
        ids = self._create_sub_agents(parent, [
            {"agent_name": "Math", "agent_description": "Solves math",
             "system_prompt": "You solve math. Be brief, one line."},
        ])
        result = assign_task_tool(
            parent,
            [{"agent_id": ids[0], "task": "What is 5+5? One number only."}],
        )
        assert "Completed" in result
        assert "completed" in result.lower()

    def test_multiple_concurrent(self):
        parent = make_real_parent()
        ids = self._create_sub_agents(parent, [
            {"agent_name": f"W{i}", "agent_description": f"Worker {i}",
             "system_prompt": "Be brief. One sentence max."}
            for i in range(3)
        ])
        start = time.time()
        result = assign_task_tool(
            parent,
            [{"agent_id": aid, "task": f"What is {i}+{i}?"} for i, aid in enumerate(ids)],
        )
        elapsed = time.time() - start
        assert "Completed 3 task" in result

    def test_background_dispatch_and_status(self):
        parent = make_real_parent()
        ids = self._create_sub_agents(parent, [
            {"agent_name": "BG", "agent_description": "Background worker",
             "system_prompt": "Be brief."},
        ])
        # Dispatch without waiting
        start = time.time()
        result = assign_task_tool(
            parent,
            [{"agent_id": ids[0], "task": "Say done. One word."}],
            wait_for_completion=False,
        )
        dispatch_time = time.time() - start
        assert "Dispatched" in result
        assert dispatch_time < 1.0

        # Poll until complete
        for _ in range(30):
            status = get_task_status_tool(parent)
            if "completed" in status:
                break
            time.sleep(1)
        assert "completed" in status

    def test_fail_fast_false_mixed(self):
        """One agent raises hard, other succeeds. Both reported."""
        parent = make_real_parent()

        ids = self._create_sub_agents(parent, [
            {"agent_name": "Good", "agent_description": "Works correctly",
             "system_prompt": "Be brief."},
        ])

        # Add an agent whose .run() raises (bypasses Agent's internal error handling)
        class CrashingAgent:
            agent_name = "Crasher"
            def run(self, task):
                raise RuntimeError("hard crash")

        parent.sub_agents["crash-id"] = {
            "agent": CrashingAgent(), "name": "Crasher", "description": "Will crash",
            "system_prompt": "test", "depth": 0, "parent_agent_id": parent.id,
        }

        result = assign_task_tool(
            parent,
            [
                {"agent_id": ids[0], "task": "Say success. One word."},
                {"agent_id": "crash-id", "task": "This will crash"},
            ],
            fail_fast=False,
        )
        assert "FAILED" in result
        assert "completed" in result.lower()

    def test_no_sub_agents_error(self):
        parent = make_real_parent()
        result = assign_task_tool(
            parent,
            [{"agent_id": "nope", "task": "anything"}],
        )
        assert "Error" in result
        assert "No sub-agents" in result

    def test_invalid_agent_id(self):
        parent = make_real_parent()
        self._create_sub_agents(parent, [
            {"agent_name": "X", "agent_description": "exists"},
        ])
        result = assign_task_tool(
            parent,
            [{"agent_id": "wrong-id", "task": "anything"}],
        )
        assert "Error" in result
        assert "not found" in result


# ── cancel_task_tool Tests ───────────────────────────────────


class TestCancelTask:
    def test_cancel_not_found(self):
        parent = make_real_parent()
        _get_task_registry(parent)
        result = cancel_task_tool(parent, "nonexistent")
        assert "not found" in result.lower()


# ── Integration ──────────────────────────────────────────────


class TestIntegration:
    def test_full_flow_create_assign_verify(self):
        """End-to-end: create sub-agents → assign tasks → real LLM results."""
        parent = make_real_parent()

        create_sub_agent_tool(parent, [
            {"agent_name": "Researcher", "agent_description": "Answers factual questions",
             "system_prompt": "You answer factual questions. One sentence max."},
            {"agent_name": "Translator", "agent_description": "Translates text"},
        ])
        ids = list(parent.sub_agents.keys())
        assert len(ids) == 2

        result = assign_task_tool(parent, [
            {"agent_id": ids[0], "task": "What is the capital of Japan? One word."},
            {"agent_id": ids[1], "task": "Translate 'hello' to French. One word."},
        ])
        assert "Completed 2 task" in result
        assert "completed" in result.lower()

    def test_system_prompt_bug_regression(self):
        """The original bug: sub-agent without system_prompt runs without crash."""
        parent = make_real_parent()
        create_sub_agent_tool(parent, [
            {"agent_name": "Greeter", "agent_description": "Greets users in Spanish"},
        ])
        aid = list(parent.sub_agents.keys())[0]
        result = assign_task_tool(parent, [
            {"agent_id": aid, "task": "Say hello in Spanish. One sentence."},
        ])
        assert "Completed" in result
        assert "FAILED" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
