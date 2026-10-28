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


def run_until_it_stops(job, task):
    """Run ``job`` on a helper thread, failing instead of hanging."""
    errors = []

    def run():
        try:
            job.run(task)
        except Exception as error:
            errors.append(error)

    runner = run_in_thread(run)
    runner.join(timeout=10)
    try:
        assert not runner.is_alive(), f"run({task!r}) never returned"
        assert errors == []
        # run() can return before the tick that called stop() has finished.
        job.thread.join(timeout=5)
        assert not job.thread.is_alive(), "the tick never ended"
    except AssertionError:
        job.stop()
        raise


def record_joins(thread, joined):
    """Make ``thread.join`` also append ``thread`` to ``joined``."""
    join = thread.join

    def recording_join(timeout=None):
        joined.append(thread)
        join(timeout)

    thread.join = recording_join


def restart_right_after_the_signal(job, task, joined):
    """Make the next stop() restart ``job`` with ``task`` right after it
    signals, as a woken run() could, and record new-thread joins."""
    signal = job._signal_stop

    def signal_then_restart():
        signal()
        job._signal_stop = signal
        job._run(task)
        record_joins(job.thread, joined)

    job._signal_stop = signal_then_restart


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


def test_a_job_that_gave_up_can_be_run_again():
    agent = FailingAgent(fail_times=3)
    job = CronJob(
        agent=agent, interval="1second", max_consecutive_errors=2
    )
    with pytest.raises(CronJobExecutionError):
        job.run("t")
    assert job.schedule.jobs == []

    returned = []
    t = run_in_thread(lambda: returned.append(job.run("t")))
    deadline = time.monotonic() + 10
    while agent.successes < 2 and time.monotonic() < deadline:
        if not t.is_alive():
            break
        time.sleep(0.05)
    job.stop()
    t.join(timeout=3)

    assert agent.successes >= 2
    assert returned, "a clean stop of the restarted run raised"


def test_the_callback_can_stop_its_own_job():
    """stop() from the callback raised "cannot join current thread" and left
    the task scheduled, so the next run() ran the old task too."""
    calls, stop_errors = [], []

    def stop_the_job(output, task, metadata):
        try:
            job.stop()
        except Exception as error:
            stop_errors.append(error)
        return output

    job = CronJob(
        agent=MockAgent(calls),
        interval="1second",
        callback=stop_the_job,
    )
    run_until_it_stops(job, "A")
    assert stop_errors == []
    assert job.schedule.jobs == []

    run_until_it_stops(job, "B")
    assert stop_errors == []
    assert calls == ["A", "B"]


def test_the_agent_can_stop_its_own_job():
    """The same failure from the agent counted the tick as a failed one."""

    def stop_the_job(task):
        job.stop()
        return task

    job = CronJob(agent=stop_the_job, interval="1second")
    run_until_it_stops(job, "A")

    assert job.last_error is None
    assert (job.execution_count, job.error_count) == (1, 0)
    assert job.schedule.jobs == []


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


def test_a_failed_execution_waits_for_the_next_interval():
    agent = FailingAgent(fail_times=99)
    job = CronJob(agent=agent, interval="1hour")
    job._interval_method("t")

    job.schedule.run_all()

    assert agent.attempts == 1
    assert job.error_count == 1
    assert job.schedule.idle_seconds > 3000


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


def test_stop_from_another_thread_waits_for_the_running_tick():
    """Only stop() on the job's own thread may skip the join."""
    entered, release = threading.Event(), threading.Event()
    events = []

    def slow(task):
        entered.set()
        released = release.wait(10)
        events.append("tick ended" if released else "tick timed out")

    def join_and_release(timeout=None):
        signalled = not job.is_running and job._stop_event.is_set()
        events.append(
            f"join(5)={timeout == 5}, signalled={signalled}"
        )
        release.set()
        join(timeout)

    def stop():
        job.stop()
        events.append("stop() returned")

    job = CronJob(agent=slow, interval="1second")
    runner = run_in_thread(lambda: job.run("t"))
    try:
        assert entered.wait(10)
        join = job.thread.join
        # The tick can only end once stop() joins, so the order is fixed.
        job.thread.join = join_and_release
        # Named like the scheduler thread: stop() must compare identities.
        stopper = threading.Thread(
            target=stop, name=job.thread.name, daemon=True
        )
        stopper.start()
        stopper.join(timeout=15)
        seen = list(events)
    finally:
        release.set()
        job.stop()
    runner.join(timeout=5)

    assert seen == [
        "join(5)=True, signalled=True",
        "tick ended",
        "stop() returned",
    ]


def test_stop_clears_a_job_that_never_started():
    """stop() only cleared the schedule of a job that had started."""
    job = CronJob(agent=MockAgent(), interval="1second")
    job.every_seconds(1, "a")
    job.every_seconds(1, "b")
    job.stop()
    assert job.schedule.jobs == []


def test_restart_after_a_timed_out_stop_retires_the_old_thread():
    entered, release = threading.Event(), threading.Event()

    def slow(task):
        entered.set()
        release.wait(10)

    job = CronJob(agent=slow, interval="1second")
    run_in_thread(lambda: job.run("t"))
    assert entered.wait(5)
    job.stop()
    old = job.thread
    assert old.is_alive()

    job.start()
    release.set()
    old.join(timeout=3)
    still_alive = old.is_alive()
    job.stop()

    assert not still_alive, "two scheduler threads share one job"


def test_a_run_restarted_during_stop_keeps_its_task():
    """Once signalled, run() can return and restart the job before
    stop() ends; it then joined the new thread and wiped its task."""
    calls, joined = [], []
    job = CronJob(agent=MockAgent(calls), interval="1hour")
    job._run("A")
    first = job.thread
    record_joins(first, joined)
    restart_right_after_the_signal(job, "B", joined)
    job.stop()
    joined_by_stop = list(joined)
    job.schedule.run_all()
    job.stop()

    assert job.thread is not first, "the restart hook never ran"
    assert joined_by_stop == [first], "stop() joined the wrong thread"
    assert calls == ["B"], "the restarted run lost its task"


def test_a_run_restarted_during_a_callback_stop_keeps_its_task():
    """The same restart while the callback stops the job: stop() runs on
    the thread it stopped, so it must not join any thread."""
    joined, stop_errors, stopped = [], [], threading.Event()

    def stop_once(output, task, metadata):
        job.callback = None
        try:
            job.stop()
        except Exception as error:
            stop_errors.append(error)
        stopped.set()
        return output

    job = CronJob(
        agent=MockAgent(), interval="1second", callback=stop_once
    )
    restart_right_after_the_signal(job, "B", joined)
    job._run("A")
    record_joins(job.thread, joined)
    try:
        assert stopped.wait(10), "the callback's stop() hung"
        left, joined_by_stop = len(job.schedule.jobs), list(joined)
    finally:
        job.stop()

    assert stop_errors == []
    assert joined_by_stop == [], "stop() joined from the callback"
    assert left == 1, "the restarted run lost its task"


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


def test_stop_many_does_not_wait_out_each_jobs_sleep():
    jobs = CronJob.run_many(
        [
            {"agent": MockAgent(), "interval": "1hour", "task": "t"}
            for _ in range(10)
        ],
        block=False,
    )

    started = time.monotonic()
    CronJob.stop_many(jobs)
    elapsed = time.monotonic() - started

    assert all(not job.thread.is_alive() for job in jobs)
    assert (
        elapsed < 1.0
    ), f"stop_many took {elapsed:.2f}s for 10 idle jobs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
