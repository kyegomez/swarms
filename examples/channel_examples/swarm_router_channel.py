"""One channel, forwarded to whichever swarm the router builds."""

from swarms import Agent, FileChannel, SwarmRouter

agents = [
    Agent(agent_name="Analyst", model_name="gpt-5.4", max_loops=1),
    Agent(agent_name="Writer", model_name="gpt-5.4", max_loops=1),
    Agent(agent_name="Reviewer", model_name="gpt-5.4", max_loops=1),
]

channel = FileChannel("runs/router.md")

# Swap swarm_type for "ConcurrentWorkflow", "GroupChat" or "MixtureOfAgents": the file stays the same.
router = SwarmRouter(
    agents=agents,
    swarm_type="SequentialWorkflow",
    max_loops=1,
    channel=channel,
)
router.run("Write a short blog post about transformer architectures.")

print(channel.path.read_text())
