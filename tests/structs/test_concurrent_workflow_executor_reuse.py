import gc
import weakref
from unittest.mock import MagicMock

import pytest

from swarms.structs.concurrent_workflow import (
    ConcurrentWorkflow,
    _shutdown_executor,
)


class FakeAgent:
    def __init__(self, agent_name="worker"):
        self.agent_name = agent_name
        self.description = "Test fake agent"
        self.history = []
        self.print_on = False

    def run(self, task, img=None, imgs=None, streaming_callback=None):
        self.history.append(task)
        if streaming_callback:
            streaming_callback(f"chunk-{task}")
        return f"result-{self.agent_name}-{task}"


def test_executor_is_initially_none():
    agent = FakeAgent("a1")
    wf = ConcurrentWorkflow(agents=[agent])
    assert getattr(wf, "_executor", "NOT_FOUND") is None
    assert getattr(wf, "_finalizer", "NOT_FOUND") is None


def test_get_executor_reuses_instance():
    agent1 = FakeAgent("a1")
    agent2 = FakeAgent("a2")
    wf = ConcurrentWorkflow(agents=[agent1, agent2])
    pool1 = wf._get_executor()
    pool2 = wf._get_executor()
    assert pool1 is pool2
    assert wf._executor is pool1
    wf.close()
    assert wf._executor is None


def test_get_executor_dynamic_upscaling():
    agent1 = FakeAgent("a1")
    wf = ConcurrentWorkflow(agents=[agent1], max_workers=2)
    pool1 = wf._get_executor(2)
    assert pool1._max_workers == 2

    # Upscaling to 6 workers should shut down pool1 and create pool2
    pool2 = wf._get_executor(6)
    assert pool2._max_workers == 6
    assert pool2 is not pool1
    assert pool1._shutdown is True

    wf.close()


def test_close_teardown():
    agent = FakeAgent("a1")
    wf = ConcurrentWorkflow(agents=[agent])
    pool = wf._get_executor()
    assert pool._shutdown is False
    wf.close()
    assert wf._executor is None
    assert wf._finalizer is None
    assert pool._shutdown is True


def test_context_manager():
    agent = FakeAgent("a1")
    with ConcurrentWorkflow(agents=[agent]) as wf:
        pool = wf._get_executor()
        assert pool._shutdown is False
    assert wf._executor is None
    assert pool._shutdown is True


def test_weakref_finalizer_cleanup():
    agent = FakeAgent("a1")
    wf = ConcurrentWorkflow(agents=[agent])
    pool = wf._get_executor()
    pool_ref = weakref.ref(pool)
    assert pool._shutdown is False

    del wf
    gc.collect()

    assert pool_ref() is not None
    assert pool_ref()._shutdown is True
    _shutdown_executor(pool_ref())


def test_executor_reused_across_multiple_runs():
    agent1 = FakeAgent("a1")
    agent2 = FakeAgent("a2")
    wf = ConcurrentWorkflow(agents=[agent1, agent2])

    res1 = wf.run("task 1")
    assert res1 is not None
    assert wf._executor is not None
    pool = wf._executor

    # Second run should reuse pool
    res2 = wf.run("task 2")
    assert res2 is not None
    assert wf._executor is pool
    assert pool._shutdown is False

    # Batch run should also reuse pool
    batch_res = wf.batch_run(["task 3", "task 4"])
    assert len(batch_res) == 2
    assert wf._executor is pool
    assert pool._shutdown is False

    wf.close()
    assert pool._shutdown is True
    assert wf._executor is None
