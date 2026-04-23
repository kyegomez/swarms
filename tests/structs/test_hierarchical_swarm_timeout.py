"""
Unit tests for HierarchicalSwarm worker timeout and retry.
No real LLM calls — all agents are patched.
"""

import time
import pytest
from unittest.mock import MagicMock, patch

from swarms.structs.hiearchical_swarm import (
    HierarchicalSwarm,
    HierarchicalOrder,
)


def _make_swarm(**kwargs) -> HierarchicalSwarm:
    agent_a = MagicMock()
    agent_a.agent_name = "AgentA"
    agent_b = MagicMock()
    agent_b.agent_name = "AgentB"

    swarm = HierarchicalSwarm.__new__(HierarchicalSwarm)
    swarm.name = "TestSwarm"
    swarm.description = "test"
    swarm.agents = [agent_a, agent_b]
    swarm.agent_map = {"AgentA": agent_a, "AgentB": agent_b}
    swarm.max_loops = 1
    swarm.output_type = "dict-all-except-first"
    swarm.director = MagicMock()
    swarm.director.agent_name = "Director"
    swarm.add_collaboration_prompt = False
    swarm.director_feedback_on = False
    swarm.interactive = False
    swarm.dashboard = None
    swarm.multi_agent_prompt_improvements = False
    swarm.director_temperature = 0.7
    swarm.director_top_p = 0.9
    swarm.planning_enabled = False
    swarm.autosave = False
    swarm.verbose = False
    swarm.parallel_execution = kwargs.get("parallel_execution", True)
    swarm.worker_timeout = kwargs.get("worker_timeout", None)
    swarm.heartbeat_interval = kwargs.get("heartbeat_interval", 30)
    swarm.max_retries = kwargs.get("max_retries", 2)
    swarm.agent_as_judge = False
    swarm.judge_agent_model_name = "gpt-5.4"
    swarm.swarm_workspace_dir = None
    swarm.feedback_director_model_name = "gpt-5.4"
    swarm.director_name = "Director"
    swarm.director_model_name = "gpt-5.4"
    swarm.director_system_prompt = ""

    from swarms.structs.conversation import Conversation

    swarm.conversation = Conversation(time_enabled=False)

    return swarm


def _orders(*pairs):
    return [HierarchicalOrder(agent_name=n, task=t) for n, t in pairs]


# ---------------------------------------------------------------------------
# Backward compat — no timeout set, original behaviour preserved
# ---------------------------------------------------------------------------


def test_no_timeout_parallel():
    swarm = _make_swarm(worker_timeout=None, parallel_execution=True)
    swarm.call_single_agent = (
        lambda name, task, cb=None, add=True: f"done:{name}"
    )
    results = swarm.execute_orders(
        _orders(("AgentA", "t1"), ("AgentB", "t2"))
    )
    assert results[0] == "done:AgentA"
    assert results[1] == "done:AgentB"


def test_no_timeout_sequential():
    swarm = _make_swarm(worker_timeout=None, parallel_execution=False)
    swarm.call_single_agent = (
        lambda name, task, cb=None, add=True: f"done:{name}"
    )
    results = swarm.execute_orders(
        _orders(("AgentA", "t1"), ("AgentB", "t2"))
    )
    assert results[0] == "done:AgentA"
    assert results[1] == "done:AgentB"


# ---------------------------------------------------------------------------
# Timeout fires and produces [FAILED] after max_retries exhausted
# ---------------------------------------------------------------------------


def test_parallel_timeout_fails():
    swarm = _make_swarm(
        worker_timeout=1, max_retries=0, parallel_execution=True
    )

    def _call(name, task, cb=None, add=True):
        if name == "AgentA":
            time.sleep(60)
        return "ok"

    swarm.call_single_agent = _call
    results = swarm.execute_orders(
        _orders(("AgentA", "slow"), ("AgentB", "fast"))
    )
    assert results[0].startswith("[FAILED]")
    assert "AgentA" in results[0]
    assert results[1] == "ok"


def test_sequential_timeout_fails():
    swarm = _make_swarm(
        worker_timeout=1, max_retries=0, parallel_execution=False
    )

    def _call(name, task, cb=None, add=True):
        if name == "AgentA":
            time.sleep(60)
        return "ok"

    swarm.call_single_agent = _call
    results = swarm.execute_orders(
        _orders(("AgentA", "slow"), ("AgentB", "fast"))
    )
    assert results[0].startswith("[FAILED]")
    assert results[1] == "ok"


# ---------------------------------------------------------------------------
# Retry — agent recovers on 3rd attempt (max_retries=2)
# ---------------------------------------------------------------------------


def test_parallel_retry_recovers():
    attempt = {"n": 0}

    def _call(name, task, cb=None, add=True):
        attempt["n"] += 1
        if attempt["n"] < 3:
            time.sleep(60)
        return "recovered"

    swarm = _make_swarm(
        worker_timeout=1, max_retries=2, parallel_execution=True
    )
    swarm.call_single_agent = _call
    results = swarm.execute_orders(_orders(("AgentA", "flaky")))
    assert results[0] == "recovered"
    assert attempt["n"] == 3


def test_sequential_retry_recovers():
    attempt = {"n": 0}

    def _call(name, task, cb=None, add=True):
        attempt["n"] += 1
        if attempt["n"] < 3:
            time.sleep(60)
        return "recovered"

    swarm = _make_swarm(
        worker_timeout=1, max_retries=2, parallel_execution=False
    )
    swarm.call_single_agent = _call
    results = swarm.execute_orders(_orders(("AgentA", "flaky")))
    assert results[0] == "recovered"
    assert attempt["n"] == 3


# ---------------------------------------------------------------------------
# Exhausted retries → correct attempt count in [FAILED] message
# ---------------------------------------------------------------------------


def test_exhausted_retries_message():
    swarm = _make_swarm(
        worker_timeout=1, max_retries=1, parallel_execution=True
    )
    swarm.call_single_agent = (
        lambda name, task, cb=None, add=True: time.sleep(60)
    )
    results = swarm.execute_orders(_orders(("AgentA", "impossible")))
    assert results[0].startswith("[FAILED]")
    assert (
        "2 attempt" in results[0]
    )  # max_retries=1 → 2 total attempts


# ---------------------------------------------------------------------------
# Healthy sibling completes even when another worker is stuck
# ---------------------------------------------------------------------------


def test_healthy_sibling_unaffected():
    swarm = _make_swarm(
        worker_timeout=1, max_retries=0, parallel_execution=True
    )

    def _call(name, task, cb=None, add=True):
        if name == "AgentB":
            time.sleep(60)
        return "fast"

    swarm.call_single_agent = _call
    results = swarm.execute_orders(
        _orders(("AgentA", "fast"), ("AgentB", "slow"))
    )
    assert results[0] == "fast"
    assert results[1].startswith("[FAILED]")


# ---------------------------------------------------------------------------
# Successful outputs written to conversation
# ---------------------------------------------------------------------------


def test_conversation_written():
    swarm = _make_swarm(
        worker_timeout=5, max_retries=0, parallel_execution=True
    )
    swarm.call_single_agent = (
        lambda name, task, cb=None, add=True: f"out:{name}"
    )
    swarm.execute_orders(_orders(("AgentA", "t1"), ("AgentB", "t2")))
    roles = [
        m["role"] for m in swarm.conversation.conversation_history
    ]
    assert "AgentA" in roles
    assert "AgentB" in roles
