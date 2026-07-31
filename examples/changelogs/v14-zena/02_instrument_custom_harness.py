"""Zena: instrument your own harness with two calls.

capture_init() in the constructor, @trace_run on the entry point.

ContextThreadPoolExecutor is the important detail: a plain
ThreadPoolExecutor drops OpenTelemetry context at the thread boundary, so
child spans detach and surface as orphans even when the caller is traced.
"""

from typing import List

from swarms import Agent
from swarms.telemetry.otel import (
    ContextThreadPoolExecutor,
    capture_init,
    trace_run,
)


class MySwarm:
    """A minimal fan-out harness that reports correctly to a collector."""

    def __init__(self, agents: List[Agent]):
        self.agents = agents
        capture_init(self)

    @trace_run("MySwarm.run")
    def run(self, task: str):
        with ContextThreadPoolExecutor(max_workers=8) as executor:
            return list(
                executor.map(lambda a: a.run(task), self.agents)
            )


if __name__ == "__main__":
    agents = [
        Agent(
            agent_name=f"Worker-{i}",
            model_name="gpt-5.4-mini",
            max_loops=1,
        )
        for i in range(3)
    ]
    print(MySwarm(agents).run("Name one risk of autonomous agents."))
