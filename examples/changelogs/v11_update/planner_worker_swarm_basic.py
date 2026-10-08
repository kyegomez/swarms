from swarms import PlannerWorkerSwarm

swarm = PlannerWorkerSwarm(
    name="ResearchSwarm",
    description="Multi-agent research and synthesis pipeline",
    director_model_name="gpt-5.4",
    worker_model_name="claude-sonnet-4-6",
    num_workers=4,
    max_loops=3,
    agent_as_judge=True,  # enable quality-gate judge cycle
)

result = swarm.run(
    "Produce a comprehensive market analysis report for EV battery technology"
)
print(result)
