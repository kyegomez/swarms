"""A planner's steps and the workers that execute them, in one file."""

from swarms import Agent, FileChannel
from swarms.structs.planner_worker_swarm import PlannerWorkerSwarm

workers = [
    Agent(
        agent_name="Market-Worker",
        agent_description="Market sizing and competitors.",
        model_name="gpt-5.4",
        max_loops=1,
    ),
    Agent(
        agent_name="Pricing-Worker",
        agent_description="Pricing and packaging.",
        model_name="gpt-5.4",
        max_loops=1,
    ),
]

channel = FileChannel("runs/planner_worker.md")

swarm = PlannerWorkerSwarm(
    agents=workers,
    planner_model_name="gpt-5.4",
    max_loops=1,
    channel=channel,
)
swarm.run("Draft a go-to-market plan for a B2B analytics product.")

print(channel.path.read_text())
