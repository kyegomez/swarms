import threading
import time
import pytest

from swarms.schemas.planner_worker_schemas import PlannerTask, PlannerTaskStatus
from swarms.structs.conversation import Conversation
from swarms.structs.planner_worker_swarm import TaskQueue, WorkerPool


class StuckAgent:
    """Agent that hangs indefinitely in run()."""

    agent_name = "stuck-worker"

    def __init__(self):
        self.block_event = threading.Event()
        self.call_count = 0

    def short_memory_init(self):
        return None

    def run(self, task=None, img=None):
        self.call_count += 1
        self.block_event.wait()
        return "unblocked"


class SlowAgent:
    """Agent that takes a long time in run()."""

    agent_name = "slow-worker"

    def __init__(self, delay=0.5):
        self.delay = delay
        self.call_count = 0

    def short_memory_init(self):
        return None

    def run(self, task=None, img=None):
        self.call_count += 1
        time.sleep(self.delay)
        return "finished"


class FastAgent:
    """Agent that finishes immediately."""

    agent_name = "fast-worker"

    def __init__(self, answer="fast success"):
        self.answer = answer
        self.call_count = 0

    def short_memory_init(self):
        return None

    def run(self, task=None, img=None):
        self.call_count += 1
        return self.answer


class ErrorAgent:
    """Agent that raises an unhandled exception."""

    agent_name = "error-worker"

    def __init__(self):
        self.call_count = 0

    def short_memory_init(self):
        return None

    def run(self, task=None, img=None):
        self.call_count += 1
        raise RuntimeError("Agent crashed during execution")


def test_worker_pool_stuck_worker_timeout_frees_pool():
    """WorkerPool with task_timeout must not hang forever when an agent gets stuck."""
    queue = TaskQueue()
    queue.add_task(PlannerTask(title="hang-task", description="stuck task", max_retries=0))

    agent = StuckAgent()
    pool = WorkerPool(
        agents=[agent],
        task_queue=queue,
        conversation=Conversation(),
        task_timeout=0.2,
    )

    runner_thread = threading.Thread(target=pool.run, daemon=True)
    runner_thread.start()
    runner_thread.join(timeout=1.5)

    try:
        assert not runner_thread.is_alive(), "pool.run() hung and did not return within 1.5s when agent was stuck"
        tasks = queue.get_all_tasks()
        assert len(tasks) == 1
        task = tasks[0]
        assert task.status == PlannerTaskStatus.FAILED
        assert "exceeded 0.2s timeout" in str(task.error)
    finally:
        agent.block_event.set()


def test_worker_pool_slow_worker_fails_fast_on_timeout():
    """WorkerPool must abandon slow execution at timeout rather than waiting for completion."""
    queue = TaskQueue()
    queue.add_task(PlannerTask(title="slow-task", description="slow task", max_retries=1))

    agent = SlowAgent(delay=0.6)
    pool = WorkerPool(
        agents=[agent],
        task_queue=queue,
        conversation=Conversation(),
        task_timeout=0.1,
    )

    start = time.time()
    pool.run()
    elapsed = time.time() - start

    assert elapsed < 0.5, f"Expected elapsed < 0.5s, took {elapsed:.2f}s (worker waited for slow run)"
    tasks = queue.get_all_tasks()
    assert len(tasks) == 1
    task = tasks[0]
    assert task.status == PlannerTaskStatus.FAILED
    assert "exceeded 0.1s timeout" in str(task.error)


def test_worker_pool_fast_agent_completes_normally():
    """WorkerPool with task_timeout completes normally when agent is fast."""
    queue = TaskQueue()
    queue.add_task(PlannerTask(title="fast-task", description="fast task", max_retries=1))

    agent = FastAgent()
    pool = WorkerPool(
        agents=[agent],
        task_queue=queue,
        conversation=Conversation(),
        task_timeout=1.0,
    )

    results = pool.run()
    tasks = queue.get_all_tasks()
    assert len(tasks) == 1
    task = tasks[0]
    assert task.status == PlannerTaskStatus.COMPLETED
    assert task.result == "fast success"
    assert results.get(task.id) == "fast success"


def test_worker_pool_without_task_timeout():
    """WorkerPool without task_timeout executes agent directly."""
    queue = TaskQueue()
    queue.add_task(PlannerTask(title="normal-task", description="normal task", max_retries=0))

    agent = FastAgent()
    pool = WorkerPool(
        agents=[agent],
        task_queue=queue,
        conversation=Conversation(),
        task_timeout=None,
    )

    results = pool.run()
    tasks = queue.get_all_tasks()
    assert len(tasks) == 1
    task = tasks[0]
    assert task.status == PlannerTaskStatus.COMPLETED
    assert task.result == "fast success"


def test_worker_pool_error_agent_handled_with_timeout():
    """WorkerPool handles unexpected agent exceptions cleanly when task_timeout is configured."""
    queue = TaskQueue()
    queue.add_task(PlannerTask(title="error-task", description="error task", max_retries=0))

    agent = ErrorAgent()
    pool = WorkerPool(
        agents=[agent],
        task_queue=queue,
        conversation=Conversation(),
        task_timeout=0.5,
    )

    pool.run()
    tasks = queue.get_all_tasks()
    assert len(tasks) == 1
    task = tasks[0]
    assert task.status == PlannerTaskStatus.FAILED
    assert "Agent crashed during execution" in str(task.error)


def test_worker_pool_mixed_tasks_stuck_and_normal():
    """WorkerPool successfully completes healthy tasks even when another task times out."""
    queue = TaskQueue()
    queue.add_task(PlannerTask(title="t-hang", description="stuck task", max_retries=0))
    queue.add_task(PlannerTask(title="t-good", description="good task", max_retries=0))

    stuck_agent = StuckAgent()
    good_agent = FastAgent(answer="success output")

    pool = WorkerPool(
        agents=[stuck_agent, good_agent],
        task_queue=queue,
        conversation=Conversation(),
        task_timeout=0.2,
        max_workers=2,
    )

    runner_thread = threading.Thread(target=pool.run, daemon=True)
    runner_thread.start()
    runner_thread.join(timeout=2.0)

    try:
        assert not runner_thread.is_alive(), "pool.run() did not finish in time"
        task_dict = {t.title: t for t in queue.get_all_tasks()}
        assert task_dict["t-good"].status == PlannerTaskStatus.COMPLETED
        assert task_dict["t-good"].result == "success output"
        assert task_dict["t-hang"].status == PlannerTaskStatus.FAILED
        assert "timeout" in str(task_dict["t-hang"].error).lower()
    finally:
        stuck_agent.block_event.set()
