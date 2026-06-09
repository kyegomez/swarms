"""
MoA worker context example.

Shows that workers receive only the original task on layer 0 and
task + previous-layer synthesis on later layers — not the full transcript.
"""

from swarms import Agent, MixtureOfAgents

workers = [
    Agent(
        agent_name=f"Worker-{i}",
        system_prompt="You are a concise research assistant.",
        model_name="claude-haiku-4-5-20251001",
        max_loops=1,
        output_type="str-all-except-first",
    )
    for i in range(3)
]

aggregator = Agent(
    agent_name="Aggregator",
    system_prompt="Synthesise the worker responses into one clear answer.",
    model_name="claude-sonnet-4-5",
    max_loops=1,
)

swarm = MixtureOfAgents(
    agents=workers,
    aggregator_agent=aggregator,
    layers=2,
    output_type="final",
)

result = swarm.run(
    "What are the main trade-offs between SQL and NoSQL databases?"
)
print(result)
