"""Zena: thread pools sized for I/O, not CPU count.

Agent calls are network-bound LLM requests, so os.cpu_count() was the wrong
signal — it under-subscribed on small machines and over-subscribed relative
to the workload. max_workers is now exposed across the concurrent harnesses.
"""

from swarms import Agent, ConcurrentWorkflow, MixtureOfAgents

workers = [
    Agent(agent_name=f"Worker-{i}", model_name="gpt-5.4", max_loops=1)
    for i in range(4)
]
aggregator = Agent(
    agent_name="Aggregator",
    system_prompt="Synthesize the expert responses into one answer.",
    model_name="gpt-5.4",
    max_loops=1,
)

TASK = (
    "What are the best practices for securing a Kubernetes cluster?"
)

print(ConcurrentWorkflow(agents=workers, max_workers=16).run(TASK))

print(
    MixtureOfAgents(
        agents=workers,
        aggregator_agent=aggregator,
        max_workers=8,
    ).run(TASK)
)
