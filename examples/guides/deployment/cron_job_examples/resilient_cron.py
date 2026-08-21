"""
Failure Handling and Monitoring

A long-running job will hit failures: a rate limit, a dropped connection, a
provider hiccup. CronJob treats those the way cron does. The failure is logged
and the task is retried on the next tick. One bad call does not stop the
schedule.

Two things let you control and observe that:

    max_consecutive_errors   stop a job that is failing *every* time, instead
                             of retrying forever. Omit it to never give up.
                             When the budget is exhausted the job stops AND
                             run() raises, so a dead schedule is never mistaken
                             for a healthy one.

    get_execution_stats()    poll from another thread while the job runs, to
                             see successes, failures and the last error.

This example runs a deliberately flaky agent so you can watch both.
"""

import random
import threading
import time

from swarms.structs.cron_job import CronJob, CronJobExecutionError


class FlakyAgent:
    """Fails roughly half the time, to stand in for an unreliable API."""

    def run(self, task: str = None, **kwargs):
        if random.random() < 0.5:
            raise ConnectionError("upstream API timed out")
        return f"ok: {task}"


def monitor(job: CronJob, every: float = 5.0) -> None:
    """Print a health line periodically while the job runs.

    Waits for the job to come up first: this thread is started before
    ``job.run()``, so ``is_running`` is still False at this point and looping
    on it directly would exit immediately.
    """
    while not job.is_running:
        time.sleep(0.1)

    while job.is_running:
        time.sleep(every)
        stats = job.get_execution_stats()
        print(
            f"  [monitor] ok={stats['execution_count']} "
            f"failed={stats['error_count']} "
            f"in-a-row={stats['consecutive_errors']} "
            f"last_error={stats['last_error']}"
        )


if __name__ == "__main__":
    job = CronJob(
        agent=FlakyAgent(),
        interval="2seconds",
        # Ten failures in a row means something is genuinely wrong, not just
        # flaky. Give up then, rather than hammering a dead endpoint forever.
        max_consecutive_errors=10,
    )

    threading.Thread(target=monitor, args=(job,), daemon=True).start()

    # Stop after a minute so the example terminates on its own.
    threading.Timer(60, job.stop).start()

    print("Flaky agent, retrying through failures. Ctrl-C to stop.\n")

    try:
        job.run("Fetch the latest reading.")
    except CronJobExecutionError as e:
        # Only reached if max_consecutive_errors was exhausted. A clean stop()
        # returns normally instead.
        print(f"\nJob gave up: {e}")
    else:
        stats = job.get_execution_stats()
        print(
            f"\nStopped cleanly after {stats['execution_count']} "
            f"successful run(s) and {stats['error_count']} failure(s)."
        )
