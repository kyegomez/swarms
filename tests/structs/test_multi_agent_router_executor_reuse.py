import gc
from unittest.mock import MagicMock
import pytest

from swarms.structs.multi_agent_router import MultiAgentRouter


class MockWorker:
    def __init__(self, name: str, description: str = "Worker agent"):
        self.agent_name = name
        self.name = name
        self.description = description

    def run(self, task: str):
        return f"{self.agent_name} processed {task}"


def test_multi_agent_router_has_executor_reuse_methods():
    router = MultiAgentRouter(agents=[MockWorker("Agent1")])
    assert hasattr(router, "_get_executor"), "MultiAgentRouter must have _get_executor"
    assert hasattr(router, "close"), "MultiAgentRouter must have close()"
    assert hasattr(router, "__enter__"), "MultiAgentRouter must support context manager"
    assert hasattr(router, "__exit__"), "MultiAgentRouter must support context manager"


def test_handle_multiple_handoffs_reuses_executor_across_runs():
    w1 = MockWorker("Agent1")
    w2 = MockWorker("Agent2")
    router = MultiAgentRouter(agents=[w1, w2])

    boss_resp = {
        "handoffs": [
            {"agent_name": "Agent1", "task": "Task A"},
            {"agent_name": "Agent2", "task": "Task B"},
        ]
    }

    router.handle_multiple_handoffs(boss_resp, "original task")
    first_executor = getattr(router, "_executor", None)
    assert first_executor is not None, "Executor must be created and cached during handle_multiple_handoffs"

    router.handle_multiple_handoffs(boss_resp, "second task")
    second_executor = getattr(router, "_executor", None)
    assert second_executor is first_executor, "Executor must be reused across consecutive calls"

    router.close()
    assert router._executor is None


def test_concurrent_batch_run_reuses_executor_across_runs():
    w1 = MockWorker("Agent1")
    router = MultiAgentRouter(agents=[w1])
    router.route_task = MagicMock(side_effect=lambda task: f"routed {task}")

    res1 = router.concurrent_batch_run(["t1", "t2"])
    assert res1 == ["routed t1", "routed t2"]
    first_executor = getattr(router, "_executor", None)
    assert first_executor is not None

    res2 = router.concurrent_batch_run(["t3", "t4"])
    assert res2 == ["routed t3", "routed t4"]
    second_executor = getattr(router, "_executor", None)
    assert second_executor is first_executor

    router.close()
    assert router._executor is None


def test_multi_agent_router_dynamic_capacity():
    w1 = MockWorker("Agent1")
    router = MultiAgentRouter(agents=[w1])

    exec1 = router._get_executor(max_workers=2)
    assert exec1._max_workers == 2

    # Smaller request reuses existing executor
    exec2 = router._get_executor(max_workers=1)
    assert exec2 is exec1

    # Larger request upscales executor
    exec3 = router._get_executor(max_workers=6)
    assert exec3 is not exec1
    assert exec3._max_workers == 6
    router.close()


def test_multi_agent_router_close_and_recreate():
    w1 = MockWorker("Agent1")
    router = MultiAgentRouter(agents=[w1])

    exec1 = router._get_executor(max_workers=2)
    assert exec1 is not None

    router.close()
    assert router._executor is None

    exec2 = router._get_executor(max_workers=2)
    assert exec2 is not None
    assert exec2 is not exec1
    router.close()


def test_multi_agent_router_context_manager():
    w1 = MockWorker("Agent1")
    with MultiAgentRouter(agents=[w1]) as router:
        exec_ref = router._get_executor(max_workers=2)
        assert exec_ref is not None

    assert router._executor is None


def test_multi_agent_router_weakref_cleanup():
    w1 = MockWorker("Agent1")
    router = MultiAgentRouter(agents=[w1])
    executor = router._get_executor(max_workers=2)
    assert executor is not None

    del router
    gc.collect()
