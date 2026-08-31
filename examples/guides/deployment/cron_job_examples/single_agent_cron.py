"""
Single Agent CronJob

The smallest useful CronJob: one agent, one interval, running until you stop it.

`run()` blocks the calling thread. The agent runs on a background thread every
interval. Press Ctrl-C to stop.

Interval format is "<number><unit>", where unit is second(s), minute(s) or
hour(s): "30seconds", "10minutes", "1hour".
"""

from swarms import Agent, CronJob

agent = Agent(
    agent_name="Market-Watcher",
    agent_description="Watches the market and reports anything notable",
    system_prompt=(
        "You are a market analyst. Report only what is notable since the "
        "last check. Three bullets maximum. If nothing is notable, say so "
        "in one line rather than padding."
    ),
    model_name="gpt-5.4",
    max_loops=1,
    print_on=True,
)

job = CronJob(agent=agent, interval="30seconds")

if __name__ == "__main__":
    print("Running every 30 seconds. Ctrl-C to stop.\n")
    job.run(
        "Summarise anything notable in the AI chip market right now."
    )
