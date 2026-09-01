"""
Tests for ``swarms.structs.batch_agent_execution.batch_agent_execution`` (#2122).

Offline: every agent here is a local stub, never an LLM. The function used to
raise on every input -- ``zip(agents, tasks, None)`` for the documented
two-argument call, and a 3-tuple unpacked into two names for the rest -- so
these cover both crashes plus the index alignment the fix relies on.
"""

import threading
import time

import pytest

from swarms.structs.agent import Agent
from swarms.structs.batch_agent_execution import (
    BatchAgentExecutionError,
    batch_agent_execution,
)


class StubAgent(Agent):
    def __init__(self, name, delay=0.0, completions=None):
        self.agent_name = name
        self.delay = delay
        self.completions = completions
        self.seen = []

    def run(self, task=None, img=None, *args, **kwargs):
        time.sleep(self.delay)
        self.seen.append((task, img))
        if self.completions is not None:
            self.completions.append(self.agent_name)
        return f"{self.agent_name}:{task}"


class BoomAgent(Agent):
    def __init__(self, name):
        self.agent_name = name

    def run(self, task=None, img=None, *args, **kwargs):
        raise RuntimeError(f"kaboom: {task}")


def test_two_argument_call_returns_results():
    agents = [StubAgent("a"), StubAgent("b")]

    results = batch_agent_execution(agents, ["task a", "task b"])

    assert results == ["a:task a", "b:task b"]


def test_imgs_are_passed_through_one_per_agent():
    agents = [StubAgent("a"), StubAgent("b")]

    batch_agent_execution(
        agents, ["task a", "task b"], ["a.png", "b.png"]
    )

    assert agents[0].seen == [("task a", "a.png")]
    assert agents[1].seen == [("task b", "b.png")]


def test_no_imgs_passes_none_to_every_agent():
    agents = [StubAgent("a"), StubAgent("b")]

    batch_agent_execution(agents, ["task a", "task b"])

    assert agents[0].seen == [("task a", None)]
    assert agents[1].seen == [("task b", None)]


def test_results_follow_input_order_not_completion_order():
    completions = []
    agents = [
        StubAgent("slow", delay=0.20, completions=completions),
        StubAgent("medium", delay=0.10, completions=completions),
        StubAgent("fast", delay=0.0, completions=completions),
    ]

    results = batch_agent_execution(
        agents, ["task 1", "task 2", "task 3"], max_workers=3
    )

    assert completions == ["fast", "medium", "slow"]
    assert results == [
        "slow:task 1",
        "medium:task 2",
        "fast:task 3",
    ]


def test_a_failing_agent_leaves_none_in_its_own_slot():
    agents = [StubAgent("a"), BoomAgent("b"), StubAgent("c")]

    results = batch_agent_execution(
        agents, ["task a", "task b", "task c"]
    )

    assert results == ["a:task a", None, "c:task c"]


def test_mismatched_task_count_is_rejected():
    agents = [StubAgent("a"), StubAgent("b")]

    with pytest.raises(BatchAgentExecutionError) as excinfo:
        batch_agent_execution(agents, ["only one task"])

    assert "Number of agents must match number of tasks" in str(
        excinfo.value
    )


def test_mismatched_img_count_is_rejected():
    agents = [StubAgent("a"), StubAgent("b")]

    with pytest.raises(BatchAgentExecutionError) as excinfo:
        batch_agent_execution(
            agents, ["task a", "task b"], ["only.png"]
        )

    assert "Number of imgs must match number of agents" in str(
        excinfo.value
    )


def test_every_agent_runs_exactly_once():
    lock = threading.Lock()
    calls = []

    class CountingAgent(StubAgent):
        def run(self, task=None, img=None, *args, **kwargs):
            with lock:
                calls.append(self.agent_name)
            return super().run(task, img, *args, **kwargs)

    agents = [CountingAgent(name) for name in ("a", "b", "c")]

    batch_agent_execution(agents, ["t1", "t2", "t3"])

    assert sorted(calls) == ["a", "b", "c"]
