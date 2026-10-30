import gc
from unittest.mock import MagicMock
import pytest

from swarms.structs.agent import Agent
from swarms.structs.agent_rearrange import AgentRearrange


def _make_agent(name="Agent-1"):
    agent = Agent.__new__(Agent)
    agent.agent_name = name
    agent.system_prompt = "test"
    agent.streaming_on = False
    agent.stream = False
    agent.run = MagicMock(return_value=f"result from {name}")
    return agent


def test_agent_rearrange_has_executor_reuse_methods():
    a1 = _make_agent("A")
    rearrange = AgentRearrange(agents=[a1], flow="A")
    assert hasattr(rearrange, "_get_executor"), "AgentRearrange must have _get_executor"
    assert hasattr(rearrange, "close"), "AgentRearrange must have close()"
    assert hasattr(rearrange, "__enter__"), "AgentRearrange must support context manager protocol"
    assert hasattr(rearrange, "__exit__"), "AgentRearrange must support context manager protocol"


def test_concurrent_workflow_reuses_executor_across_runs():
    a1 = _make_agent("A")
    a2 = _make_agent("B")
    a3 = _make_agent("C")

    rearrange = AgentRearrange(
        agents=[a1, a2, a3],
        flow="A, B -> C",
        max_loops=1,
    )

    rearrange.run("Task 1")
    first_executor = getattr(rearrange, "_executor", None)
    assert first_executor is not None, "Executor must be created and cached during concurrent run"

    rearrange.run("Task 2")
    second_executor = getattr(rearrange, "_executor", None)
    assert second_executor is first_executor, "Executor must be reused across consecutive runs"
    rearrange.close()


def test_batch_run_reuses_executor():
    a1 = _make_agent("A")
    a2 = _make_agent("B")

    rearrange = AgentRearrange(
        agents=[a1, a2],
        flow="A -> B",
        max_loops=1,
    )

    rearrange.batch_run(["Task 1", "Task 2"])
    first_executor = getattr(rearrange, "_executor", None)
    assert first_executor is not None, "Executor must be created and cached during batch_run"

    rearrange.batch_run(["Task 3", "Task 4"])
    second_executor = getattr(rearrange, "_executor", None)
    assert second_executor is first_executor, "Executor must be reused across consecutive batch_run calls"
    rearrange.close()


def test_dynamic_capacity_scaling():
    a1 = _make_agent("A")
    rearrange = AgentRearrange(agents=[a1], flow="A")

    exec1 = rearrange._get_executor(max_workers=2)
    assert exec1._max_workers == 2

    # Smaller request reuses existing executor
    exec2 = rearrange._get_executor(max_workers=1)
    assert exec2 is exec1

    # Larger request upscales executor
    exec3 = rearrange._get_executor(max_workers=5)
    assert exec3 is not exec1
    assert exec3._max_workers == 5
    rearrange.close()


def test_close_and_recreate():
    a1 = _make_agent("A")
    a2 = _make_agent("B")
    rearrange = AgentRearrange(agents=[a1, a2], flow="A, B")

    rearrange.run("Task 1")
    exec1 = rearrange._executor
    assert exec1 is not None

    rearrange.close()
    assert rearrange._executor is None

    rearrange.run("Task 2")
    exec2 = rearrange._executor
    assert exec2 is not None
    assert exec2 is not exec1
    rearrange.close()


def test_context_manager():
    a1 = _make_agent("A")
    a2 = _make_agent("B")

    with AgentRearrange(agents=[a1, a2], flow="A, B") as rearrange:
        rearrange.run("Task 1")
        assert rearrange._executor is not None

    assert rearrange._executor is None


def test_weakref_cleanup():
    a1 = _make_agent("A")
    rearrange = AgentRearrange(agents=[a1], flow="A")
    executor = rearrange._get_executor(max_workers=2)
    assert executor is not None

    del rearrange
    gc.collect()
