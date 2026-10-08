"""
Non-Blocking Fleet Control

`CronJob.run_many(..., block=False)` starts every job and returns immediately,
handing back the job objects. Use this when the schedule is not the main thing
your process does: a web server, a bot, a notebook.

You own the lifecycle from that point. Stop them with `CronJob.stop_many`.

This example starts three agents, lets them run, inspects them mid-flight, and
shuts them down.
"""

import time

from swarms.structs.cron_job import CronJob


class EchoAgent:
    """Stands in for a real Agent so the example runs without API keys."""

    def __init__(self, name: str):
        self.name = name
        self.runs = 0

    def run(self, task: str = None, **kwargs):
        self.runs += 1
        print(f"  [{self.name}] run #{self.runs}: {task}")
        return f"{self.name}:{self.runs}"


if __name__ == "__main__":
    agents = {
        "fast": EchoAgent("fast"),
        "medium": EchoAgent("medium"),
        "slow": EchoAgent("slow"),
    }

    jobs = CronJob.run_many(
        [
            {
                "agent": agents["fast"],
                "interval": "2seconds",
                "task": "poll",
                "job_id": "fast-poller",
            },
            {
                "agent": agents["medium"],
                "interval": "5seconds",
                "task": "check",
                "job_id": "medium-checker",
            },
            {
                "agent": agents["slow"],
                "interval": "10seconds",
                "task": "summarise",
                "job_id": "slow-summariser",
            },
        ],
        block=False,  # start them, hand control back
    )

    print("Three jobs started. Doing other work for 20 seconds...\n")
    time.sleep(20)

    print("\nMid-flight status:")
    for job in jobs:
        stats = job.get_execution_stats()
        print(
            f"  {stats['job_id']:<18} every {stats['interval']:<10} "
            f"ok={stats['execution_count']:<3} "
            f"failed={stats['error_count']:<3} "
            f"up={stats['uptime']:.0f}s"
        )

    CronJob.stop_many(jobs)
    print("\nAll jobs stopped.")
