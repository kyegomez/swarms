"""A director's plan and each worker's report, under their own names."""

from swarms import Agent, FileChannel, HierarchicalSwarm

workers = [
    Agent(
        agent_name="DataWorker",
        agent_description="Finds and summarises data.",
        model_name="gpt-5.4",
        max_loops=1,
    ),
    Agent(
        agent_name="WritingWorker",
        agent_description="Writes clear prose from notes.",
        model_name="gpt-5.4",
        max_loops=1,
    ),
]

channel = FileChannel("runs/hierarchical.md")

swarm = HierarchicalSwarm(
    agents=workers,
    director_model_name="gpt-5.4",
    max_loops=1,
    channel=channel,
)
swarm.run(
    "Produce a short competitive analysis of the AI chip market."
)

print(channel.path.read_text())
