from swarms import HeavySwarm


swarm = HeavySwarm(
    name="ResearchSwarm",
    description="Five-agent research/analysis/synthesis pipeline",
    worker_model_name="gpt-4.1",
    question_agent_model_name="gpt-4.1",
    max_loops=1,
    show_dashboard=True,
)

result = swarm.run(
    "What are the most promising near-term applications of small "
    "language models (under 10B parameters)?"
)

print(result)
