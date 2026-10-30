import gc
import pytest
from unittest.mock import MagicMock
from swarms import Agent
from swarms.schemas.hs_schemas import HierarchicalOrder
from swarms.structs.hiearchical_swarm import HierarchicalSwarm


def _create_mock_agent(name: str = "Worker-1") -> Agent:
    agent = Agent(
        agent_name=name,
        agent_description=f"Description for {name}",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )
    agent.run = MagicMock(return_value="result from worker")
    return agent


def test_hierarchical_swarm_has_executor_reuse_methods():
    worker = _create_mock_agent()
    swarm = HierarchicalSwarm(
        name="Test-Swarm",
        agents=[worker],
        parallel_execution=True,
    )
    assert hasattr(swarm, "_get_executor"), "HierarchicalSwarm must have _get_executor"
    assert hasattr(swarm, "close"), "HierarchicalSwarm must have close()"
    assert hasattr(swarm, "__enter__"), "HierarchicalSwarm must support context manager"
    assert hasattr(swarm, "__exit__"), "HierarchicalSwarm must support context manager"


def test_hierarchical_swarm_reuses_executor_across_runs():
    worker = _create_mock_agent("Worker-1")
    swarm = HierarchicalSwarm(
        name="Test-Swarm",
        agents=[worker],
        parallel_execution=True,
        max_workers=2,
    )

    orders = [HierarchicalOrder(agent_name="Worker-1", task="Task 1")]

    results1, failures1 = swarm._execute_orders_once(orders)
    first_executor = getattr(swarm, "_executor", None)
    assert first_executor is not None, "Executor should be created and retained after _execute_orders_once"
    assert len(results1) == 1
    assert failures1 == []

    results2, failures2 = swarm._execute_orders_once(orders)
    second_executor = getattr(swarm, "_executor", None)
    assert second_executor is first_executor, "Executor must be reused across consecutive calls"

    swarm.close()
    assert swarm._executor is None


def test_hierarchical_swarm_get_executor_dynamic_capacity():
    worker = _create_mock_agent()
    swarm = HierarchicalSwarm(
        name="Test-Swarm",
        agents=[worker],
        parallel_execution=True,
        max_workers=2,
    )
    exec1 = swarm._get_executor(max_workers=2)
    assert exec1._max_workers == 2

    # Smaller or equal capacity does not recreate
    exec2 = swarm._get_executor(max_workers=1)
    assert exec2 is exec1

    # Larger capacity expands and replaces executor
    exec3 = swarm._get_executor(max_workers=6)
    assert exec3 is not exec1
    assert exec3._max_workers == 6
    swarm.close()


def test_hierarchical_swarm_close_and_recreate():
    worker = _create_mock_agent()
    swarm = HierarchicalSwarm(
        name="Test-Swarm",
        agents=[worker],
        parallel_execution=True,
        max_workers=2,
    )
    orders = [HierarchicalOrder(agent_name="Worker-1", task="Task 1")]

    swarm._execute_orders_once(orders)
    exec1 = swarm._executor
    assert exec1 is not None

    swarm.close()
    assert swarm._executor is None

    swarm._execute_orders_once(orders)
    exec2 = swarm._executor
    assert exec2 is not None
    assert exec2 is not exec1
    swarm.close()


def test_hierarchical_swarm_context_manager():
    worker = _create_mock_agent()
    orders = [HierarchicalOrder(agent_name="Worker-1", task="Task 1")]

    with HierarchicalSwarm(
        name="Test-Swarm",
        agents=[worker],
        parallel_execution=True,
        max_workers=2,
    ) as swarm:
        swarm._execute_orders_once(orders)
        exec_ref = swarm._executor
        assert exec_ref is not None

    assert swarm._executor is None


def test_hierarchical_swarm_weakref_cleanup():
    worker = _create_mock_agent()
    swarm = HierarchicalSwarm(
        name="Test-Swarm",
        agents=[worker],
        parallel_execution=True,
        max_workers=2,
    )
    executor = swarm._get_executor(max_workers=2)
    assert executor is not None

    del swarm
    gc.collect()
