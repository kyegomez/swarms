import gc
from unittest.mock import MagicMock
import pytest

from swarms.structs.graph_workflow import GraphWorkflow


def _mock_agent(name: str, response: str = None):
    a = MagicMock()
    a.agent_name = name
    a.run = MagicMock(return_value=response or f"output-{name}")
    return a


def test_graph_workflow_has_executor_reuse_methods():
    wf = GraphWorkflow(name="test")
    assert hasattr(wf, "_get_executor"), "GraphWorkflow must have _get_executor"
    assert hasattr(wf, "close"), "GraphWorkflow must have close()"
    assert hasattr(wf, "__enter__"), "GraphWorkflow must support context manager protocol"
    assert hasattr(wf, "__exit__"), "GraphWorkflow must support context manager protocol"


def test_graph_workflow_reuses_executor_across_runs():
    wf = GraphWorkflow(name="test-reuse")
    wf.add_node(_mock_agent("a"))
    wf.add_node(_mock_agent("b"))

    wf.run("task 1")
    first_executor = getattr(wf, "_executor", None)
    assert first_executor is not None, "Executor must be created and cached in self._executor during run"

    wf.run("task 2")
    second_executor = getattr(wf, "_executor", None)
    assert second_executor is first_executor, "Executor must be reused across consecutive run calls"
    wf.close()


def test_graph_workflow_executor_dynamic_upscaling():
    wf = GraphWorkflow(name="test-scaling", max_parallel_nodes=2)

    exec1 = wf._get_executor(max_workers=2)
    assert exec1._max_workers == 2

    # Smaller request reuses existing executor
    exec2 = wf._get_executor(max_workers=1)
    assert exec2 is exec1

    # Larger request upscales executor
    exec3 = wf._get_executor(max_workers=5)
    assert exec3 is not exec1
    assert exec3._max_workers == 5
    wf.close()


def test_graph_workflow_close_and_recreate():
    wf = GraphWorkflow(name="test-close")
    wf.add_node(_mock_agent("a"))

    wf.run("task 1")
    exec1 = wf._executor
    assert exec1 is not None

    wf.close()
    assert wf._executor is None

    wf.run("task 2")
    exec2 = wf._executor
    assert exec2 is not None
    assert exec2 is not exec1
    wf.close()


def test_graph_workflow_context_manager():
    with GraphWorkflow(name="test-ctx") as wf:
        wf.add_node(_mock_agent("a"))
        wf.run("task 1")
        assert wf._executor is not None

    assert wf._executor is None


def test_graph_workflow_weakref_cleanup():
    wf = GraphWorkflow(name="test-weakref", max_parallel_nodes=2)
    executor = wf._get_executor(max_workers=2)
    assert executor is not None

    del wf
    gc.collect()
