"""Workers answer independently; the aggregator's synthesis is the last turn in the file."""

from swarms import Agent, FileChannel, MixtureOfAgents

workers = [
    Agent(
        agent_name="Security-Expert",
        system_prompt="Answer from a security point of view.",
        model_name="gpt-5.4",
        max_loops=1,
    ),
    Agent(
        agent_name="Cost-Expert",
        system_prompt="Answer from a cost point of view.",
        model_name="gpt-5.4",
        max_loops=1,
    ),
]
aggregator = Agent(
    agent_name="Aggregator",
    system_prompt="Merge the expert answers into one recommendation.",
    model_name="gpt-5.4",
    max_loops=1,
)

channel = FileChannel("runs/mixture.md")

moa = MixtureOfAgents(
    agents=workers,
    aggregator_agent=aggregator,
    layers=1,
    channel=channel,
)
moa.run(
    "Should we self-host our vector database or use a managed service?"
)

print(channel.path.read_text())
