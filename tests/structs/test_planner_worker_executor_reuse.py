import threading
import time
from unittest.mock import MagicMock, patch
import pytest

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.schemas.planner_worker_schemas import PlannerTask
from swarms.structs.planner_worker_swarm import (
    PlannerWorkerSwarm,
    TaskQueue,
    WorkerPool,
)


def _make_agent(name="worker-1"):
    agent = Agent.__new__(Agent)
    agent.agent_name = name
    agent.short_memory = None
    agent.short_memory_init = lambda: None
    agent.run = MagicMock(return_value="done")
    return agent


class TestWorkerPoolExecutorReuse:
    def test_worker_pool_reuses_executor_across_runs(self):
        agent = _make_agent("w1")
        queue = TaskQueue()
        queue.add_task(PlannerTask(id="t1", title="task 1", description="desc 1"))
        convo = Conversation()

        pool = WorkerPool(
            agents=[agent],
            task_queue=queue,
            conversation=convo,
            max_workers=2,
        )

        pool.run()
        exec1 = getattr(pool, "_executor", None)
        assert exec1 is not None, "WorkerPool._executor should be set after run()"

        queue.add_task(PlannerTask(id="t2", title="task 2", description="desc 2"))
        pool.run()
        assert pool._executor is exec1, "WorkerPool should reuse its executor across runs"
        pool.close()

    def test_worker_pool_reuses_task_executor_for_timeouts(self):
        agent = _make_agent("w1")
        queue = TaskQueue()
        queue.add_task(PlannerTask(id="t1", title="task 1", description="desc 1"))
        convo = Conversation()

        pool = WorkerPool(
            agents=[agent],
            task_queue=queue,
            conversation=convo,
            max_workers=2,
            task_timeout=10.0,
        )

        pool.run()
        task_exec1 = getattr(pool, "_task_executor", None)
        assert task_exec1 is not None, "WorkerPool._task_executor should be set when task_timeout is configured"

        queue.add_task(PlannerTask(id="t2", title="task 2", description="desc 2"))
        pool.run()
        assert pool._task_executor is task_exec1, "WorkerPool should reuse _task_executor across task executions"
        pool.close()

    def test_worker_pool_close_shuts_down_and_later_call_recreates(self):
        agent = _make_agent("w1")
        queue = TaskQueue()
        queue.add_task(PlannerTask(id="t1", title="task 1", description="desc 1"))
        convo = Conversation()

        pool = WorkerPool(
            agents=[agent],
            task_queue=queue,
            conversation=convo,
            max_workers=2,
            task_timeout=10.0,
        )

        pool.run()
        exec1 = pool._executor
        assert exec1 is not None

        pool.close()
        assert pool._executor is None
        assert pool._task_executor is None

        queue.add_task(PlannerTask(id="t2", title="task 2", description="desc 2"))
        pool.run()
        assert pool._executor is not None
        assert pool._executor is not exec1
        pool.close()

    def test_worker_pool_context_manager(self):
        agent = _make_agent("w1")
        queue = TaskQueue()
        queue.add_task(PlannerTask(id="t1", title="task 1", description="desc 1"))
        convo = Conversation()

        pool = WorkerPool(
            agents=[agent],
            task_queue=queue,
            conversation=convo,
            max_workers=2,
        )

        with pool:
            pool.run()
            assert pool._executor is not None

        assert pool._executor is None

    def test_planner_worker_swarm_close_cleans_up_worker_pool(self):
        agent = _make_agent("w1")
        swarm = PlannerWorkerSwarm(agents=[agent], max_loops=1)
        swarm._worker_pool = WorkerPool(
            agents=[agent],
            task_queue=swarm.task_queue,
            conversation=swarm.conversation,
            max_workers=2,
        )
        swarm._worker_pool.run()
        assert swarm._worker_pool._executor is not None

        swarm.close()
        assert swarm._worker_pool is None

    def test_planner_worker_swarm_context_manager(self):
        agent = _make_agent("w1")
        with PlannerWorkerSwarm(agents=[agent], max_loops=1) as swarm:
            swarm._worker_pool = WorkerPool(
                agents=[agent],
                task_queue=swarm.task_queue,
                conversation=swarm.conversation,
                max_workers=2,
            )
            swarm._worker_pool.run()
            assert swarm._worker_pool._executor is not None

        assert swarm._worker_pool is None
