"""
One Agent, Several Tasks, One Cadence

`batched_run` schedules every task in the list at the job's interval, so all of
them run on each tick. This is one agent doing several things on one schedule.

Use this when the tasks share a cadence. When they need *different* cadences,
use `CronJob.run_many` instead (see multi_agent_schedules_cron.py).
"""

from swarms import Agent, CronJob

agent = Agent(
    agent_name="Ops-Monitor",
    agent_description="Runs the recurring operational checks",
    system_prompt=(
        "You are an operations monitor. Answer the specific check you are "
        "given. Be terse: state the finding and whether it needs attention."
    ),
    model_name="gpt-5.4",
    max_loops=1,
    print_on=True,
)

CHECKS = [
    "Check whether inventory levels are below reorder thresholds.",
    "Check whether refund volume is above its weekly average.",
    "Check whether any support queue has waited longer than an hour.",
]

if __name__ == "__main__":
    print(f"{len(CHECKS)} checks every 15 minutes. Ctrl-C to stop.\n")
    CronJob(agent=agent, interval="15minutes").batched_run(CHECKS)
