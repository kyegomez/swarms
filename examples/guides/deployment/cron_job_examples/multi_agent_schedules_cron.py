"""
Multiple Agents, Different Schedules

A CronJob binds one agent to one interval, so a fleet on mixed cadences needs
one job per agent. `CronJob.run_many` builds them, starts them together, and
blocks once.

Each job gets its own scheduler thread, so the agents are isolated: a slow or
failing agent does not delay or stop the others, and each carries its own
error budget.

Here: a price checker every 30 seconds, an anomaly scan every 10 minutes, and
an hourly digest, all in one process.
"""

from swarms import Agent, CronJob


def build_agent(name: str, description: str, prompt: str) -> Agent:
    return Agent(
        agent_name=name,
        agent_description=description,
        system_prompt=prompt,
        model_name="gpt-5.4",
        max_loops=1,
        print_on=True,
    )


price_agent = build_agent(
    "Price-Checker",
    "Checks prices frequently and flags large moves",
    "Report the current price and flag any move over 2%. Two lines maximum.",
)

anomaly_agent = build_agent(
    "Anomaly-Scanner",
    "Looks for unusual patterns on a slower cadence",
    "Scan for unusual patterns. Report only genuine anomalies, not noise.",
)

digest_agent = build_agent(
    "Hourly-Digest",
    "Summarises the hour for a human reader",
    "Write a short digest of the last hour. Lead with what changed.",
)

if __name__ == "__main__":
    print("Three agents, three cadences. Ctrl-C to stop.\n")

    CronJob.run_many(
        [
            {
                "agent": price_agent,
                "interval": "30seconds",
                "task": "Check the BTC price.",
            },
            {
                "agent": anomaly_agent,
                "interval": "10minutes",
                "task": "Scan recent market data for anomalies.",
                # This one talks to a flaky data source, so give it a budget:
                # five failures in a row and it stops rather than retrying
                # forever. The other two are unaffected either way.
                "max_consecutive_errors": 5,
            },
            {
                "agent": digest_agent,
                "interval": "1hour",
                "task": "Summarise the last hour of market activity.",
            },
        ]
    )
