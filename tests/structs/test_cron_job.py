import threading
import time

import pytest

from swarms.structs.cron_job import (
    CronJob,
    CronJobConfigError,
    CronJobExecutionError,
)


class MockAgent:
    """An agent-shaped object: exposes run(), like a real Agent.

    It deliberately does *not* define __call__. CronJob dispatches on having a
    run() method, so this is the shape that used to fail before the check was
    corrected.
    """

    def __init__(self, calls_list=None):
        self.calls_list = calls_list if calls_list is not None else []

    def run(self, task: str = None, **kwargs):
        self.calls_list.append(task)
        return f"result-{task}"


class FailingAgent:
    """Fails its first ``fail_times`` calls, then succeeds."""

    def __init__(self, fail_times: int = 1):
        self.fail_times = fail_times
        self.attempts = 0
        self.successes = 0

    def run(self, task: str = None, **kwargs):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise ValueError("transient failure")
        self.successes += 1
        return f"result-{task}"


def run_in_thread(fn):
    """Start ``fn`` on a daemon thread and hand back the thread."""
    thread = threading.Thread(target=fn, daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


def test_batched_run_schedules_every_task():
    """batched_run used to block on the first task, leaving the rest unscheduled."""
    calls = []
    job = CronJob(agent=MockAgent(calls), interval="1second")

    t = run_in_thread(
        lambda: job.batched_run(["task-A", "task-B", "task-C"])
    )
    time.sleep(2.5)
    job.stop()
    t.join(timeout=2)

    assert {"task-A", "task-B", "task-C"} <= set(calls)


def test_batched_run_returns_one_job_per_task():
    job = CronJob(agent=MockAgent(), interval="1second")
    results = []

    t = run_in_thread(
        lambda: results.append(
            job.batched_run(["task-1", "task-2", "task-3"])
        )
    )
    time.sleep(0.5)
    job.stop()
    t.join(timeout=2)

    assert len(results) == 1
    assert len(results[0]) == 3


def test_run_blocks_and_returns_the_scheduled_job():
    job = CronJob(agent=MockAgent(), interval="1second")
    results = []

    t = run_in_thread(lambda: results.append(job.run("task-single")))
    time.sleep(0.5)
    job.stop()
    t.join(timeout=2)

    assert len(results) == 1
    assert results[0] is not None


def test_task_fires_repeatedly_on_its_interval():
    calls = []
    job = CronJob(agent=MockAgent(calls), interval="1second")

    t = run_in_thread(lambda: job.run("tick"))
    time.sleep(3.2)
    job.stop()
    t.join(timeout=2)

    # ~3 firings in 3.2s at a 1s interval; allow slack for a loaded machine.
    assert 2 <= len(calls) <= 4


# ---------------------------------------------------------------------------
# Agent dispatch
# ---------------------------------------------------------------------------


def test_dispatch_prefers_run_over_calling_the_agent():
    """An object with run() and no __call__ must go through run()."""
    calls = []
    job = CronJob(agent=MockAgent(calls), interval="1second")
    assert job._run_job("t") == "result-t"
    assert calls == ["t"]


def test_dispatch_falls_back_to_calling_a_plain_function():
    """isinstance(x, Callable) is true for functions, so this used to take the
    run() branch and raise AttributeError."""
    seen = []

    def plain_function(task, **kwargs):
        seen.append(task)
        return f"fn-{task}"

    job = CronJob(agent=plain_function, interval="1second")
    assert job._run_job("t") == "fn-t"
    assert seen == ["t"]


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_a_failed_execution_does_not_kill_the_schedule():
    """The loop used to set is_running=False and re-raise on the first error,
    so one transient failure stopped the job permanently."""
    agent = FailingAgent(fail_times=1)
    job = CronJob(agent=agent, interval="1second")

    t = run_in_thread(lambda: job.run("t"))
    time.sleep(3.5)
    still_running = job.is_running
    job.stop()
    t.join(timeout=3)

    assert (
        agent.successes >= 1
    ), "job never recovered from one failure"
    assert still_running, "schedule died on a transient error"
    assert job.error_count == 1
    assert job.consecutive_errors == 0, "counter not reset on success"


def test_consecutive_errors_stop_the_job_when_a_budget_is_set():
    agent = FailingAgent(fail_times=99)
    job = CronJob(
        agent=agent, interval="1second", max_consecutive_errors=2
    )

    with pytest.raises(CronJobExecutionError, match="consecutive"):
        job.run("t")

    assert job._stopped_due_to_error is True
    assert job.is_running is False


def test_without_a_budget_the_job_retries_indefinitely():
    agent = FailingAgent(fail_times=99)
    job = CronJob(agent=agent, interval="1second")

    t = run_in_thread(lambda: job.run("t"))
    time.sleep(3.5)
    alive = job.thread.is_alive()
    job.stop()
    t.join(timeout=3)

    assert agent.attempts > 1, "gave up after the first failure"
    assert alive, "scheduler thread died"


def test_stats_expose_the_failure_picture():
    agent = FailingAgent(fail_times=99)
    job = CronJob(agent=agent, interval="1second")

    t = run_in_thread(lambda: job.run("t"))
    time.sleep(2.5)
    stats = job.get_execution_stats()
    job.stop()
    t.join(timeout=3)

    assert stats["error_count"] >= 1
    assert stats["consecutive_errors"] >= 1
    assert "transient failure" in stats["last_error"]
    assert stats["stopped_due_to_error"] is False


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "interval",
    [
        "1second",
        "5seconds",
        "1minute",
        "10minutes",
        "1hour",
        "2hours",
    ],
)
def test_valid_intervals_are_accepted(interval):
    CronJob(agent=MockAgent(), interval=interval)


@pytest.mark.parametrize(
    "interval",
    [
        "0second",
        "0minutes",
        "",
        "   ",
        "-1second",
        "1day",
        "second",
        "1 second",
    ],
)
def test_invalid_intervals_are_rejected_at_construction(interval):
    """'0second' scheduled a job that silently never fired, and '' failed much
    later with a message claiming no interval had been provided."""
    with pytest.raises(CronJobConfigError):
        CronJob(agent=MockAgent(), interval=interval)


def test_missing_agent_is_rejected():
    with pytest.raises(CronJobConfigError, match="Agent"):
        CronJob(agent=None, interval="1second")


def test_hourly_interval_schedules_without_error():
    """The hour/hours lambdas took one argument while the caller passed two."""
    job = CronJob(agent=MockAgent(), interval="1hour")
    assert job._run("task-hourly") is not None
    job.stop()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_stop_joins_the_scheduler_thread():
    job = CronJob(agent=MockAgent(), interval="1second")

    t = run_in_thread(lambda: job.run("t"))
    time.sleep(1.2)
    job.stop()
    t.join(timeout=3)

    assert job.is_running is False
    leftover = [
        thread.name
        for thread in threading.enumerate()
        if thread.name.startswith(f"cronjob_{job.job_id}")
    ]
    assert leftover == []


def test_callback_customizes_output_and_a_broken_one_is_survivable():
    job = CronJob(
        agent=MockAgent(),
        interval="1second",
        callback=lambda output, task, metadata: f"wrapped({output})",
    )
    assert job._run_job("t") == "wrapped(result-t)"

    def broken_callback(output, task, metadata):
        raise RuntimeError("callback exploded")

    job.set_callback(broken_callback)
    # The original output survives a callback that raises.
    assert job._run_job("t") == "result-t"


# ---------------------------------------------------------------------------
# run_many: several agents, several cadences
# ---------------------------------------------------------------------------


def test_run_many_runs_each_agent_on_its_own_cadence():
    fast, slow = MockAgent(), MockAgent()
    jobs = CronJob.run_many(
        [
            {"agent": fast, "interval": "1second", "task": "fast"},
            {"agent": slow, "interval": "3seconds", "task": "slow"},
        ],
        block=False,
    )
    time.sleep(6.2)
    CronJob.stop_many(jobs)

    assert len(fast.calls_list) > len(slow.calls_list)
    assert len(slow.calls_list) >= 1
    assert len({job.job_id for job in jobs}) == 2


def test_run_many_isolates_a_failing_agent_from_the_others():
    healthy = MockAgent()
    broken = FailingAgent(fail_times=99)
    jobs = CronJob.run_many(
        [
            {"agent": healthy, "interval": "1second", "task": "ok"},
            {"agent": broken, "interval": "1second", "task": "bad"},
        ],
        block=False,
    )
    time.sleep(3.5)
    healthy_job, broken_job = jobs
    CronJob.stop_many(jobs)

    assert len(healthy.calls_list) >= 2, "healthy agent was held back"
    assert healthy_job.error_count == 0
    assert broken_job.error_count >= 1


def test_run_many_applies_error_budgets_per_job():
    healthy = MockAgent()
    broken = FailingAgent(fail_times=99)
    jobs = CronJob.run_many(
        [
            {"agent": healthy, "interval": "1second", "task": "ok"},
            {
                "agent": broken,
                "interval": "1second",
                "task": "bad",
                "max_consecutive_errors": 2,
            },
        ],
        block=False,
    )
    time.sleep(4.5)
    healthy_job, broken_job = jobs
    still_up = healthy_job.is_running
    CronJob.stop_many(jobs)

    assert broken_job._stopped_due_to_error is True
    assert still_up, "a sibling giving up stopped the healthy job"


def test_run_many_blocking_form_raises_when_a_job_gives_up():
    with pytest.raises(
        CronJobExecutionError, match="repeated failures"
    ):
        CronJob.run_many(
            [
                {
                    "agent": FailingAgent(fail_times=99),
                    "interval": "1second",
                    "task": "bad",
                    "max_consecutive_errors": 2,
                }
            ],
            block=True,
        )


def test_run_many_stop_many_leaves_no_threads_behind():
    jobs = CronJob.run_many(
        [
            {
                "agent": MockAgent(),
                "interval": "1second",
                "task": "a",
            },
            {
                "agent": MockAgent(),
                "interval": "1second",
                "task": "b",
            },
        ],
        block=False,
    )
    time.sleep(1.2)
    CronJob.stop_many(jobs)
    time.sleep(0.3)

    assert all(not job.is_running for job in jobs)
    ids = {job.job_id for job in jobs}
    leftover = [
        thread.name
        for thread in threading.enumerate()
        if any(thread.name == f"cronjob_{jid}" for jid in ids)
    ]
    assert leftover == []


def test_run_many_forwards_per_job_kwargs():
    received = {}

    class KwargAgent:
        def run(self, task=None, **kwargs):
            received.update(kwargs)
            return "ok"

    jobs = CronJob.run_many(
        [
            {
                "agent": KwargAgent(),
                "interval": "1second",
                "task": "t",
                "kwargs": {"img": "chart.png"},
            }
        ],
        block=False,
    )
    time.sleep(1.5)
    CronJob.stop_many(jobs)

    assert received.get("img") == "chart.png"


def test_run_many_rejects_an_empty_schedule_list():
    with pytest.raises(CronJobConfigError, match="at least one"):
        CronJob.run_many([], block=False)


@pytest.mark.parametrize(
    "spec,missing",
    [
        ({"interval": "1second", "task": "t"}, "agent"),
        ({"agent": "x", "task": "t"}, "interval"),
        ({"agent": "x", "interval": "1second"}, "task"),
    ],
)
def test_run_many_names_the_missing_key(spec, missing):
    with pytest.raises(CronJobConfigError, match=missing):
        CronJob.run_many([spec], block=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
