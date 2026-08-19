import threading
import time
from swarms.structs.cron_job import CronJob


class MockAgent:
    def __init__(self, calls_list=None):
        self.calls_list = calls_list if calls_list is not None else []

    def run(self, task: str = None, **kwargs):
        self.calls_list.append(task)
        return f"result-{task}"

    def __call__(self, task: str = None, **kwargs):
        return self.run(task=task, **kwargs)


def test_cron_job_batched_run_schedules_all_tasks():
    calls = []
    agent = MockAgent(calls)
    job = CronJob(agent=agent, interval="1second")

    t = threading.Thread(
        target=lambda: job.batched_run(["task-A", "task-B", "task-C"]),
        daemon=True,
    )
    t.start()
    time.sleep(2.5)
    job.stop()
    t.join(timeout=2)

    assert "task-A" in calls
    assert "task-B" in calls
    assert "task-C" in calls


def test_cron_job_batched_run_returns_list_of_jobs():
    calls = []
    agent = MockAgent(calls)
    job = CronJob(agent=agent, interval="1second")

    results = []

    def runner():
        res = job.batched_run(["task-1", "task-2", "task-3"])
        results.append(res)

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    time.sleep(0.5)
    job.stop()
    t.join(timeout=2)

    assert len(results) == 1
    assert len(results[0]) == 3


def test_cron_job_single_run_blocks_and_returns_job():
    calls = []
    agent = MockAgent(calls)
    job = CronJob(agent=agent, interval="1second")

    results = []

    def runner():
        res = job.run("task-single")
        results.append(res)

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    time.sleep(0.5)
    job.stop()
    t.join(timeout=2)

    assert len(results) == 1
    assert results[0] is not None
