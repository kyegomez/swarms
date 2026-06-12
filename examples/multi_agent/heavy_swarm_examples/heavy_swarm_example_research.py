from swarms import HeavySwarm


swarm = HeavySwarm(
    name="ResearchSwarm",
    description="Five-agent research/analysis/synthesis pipeline",
    worker_model_name="gpt-5.4",
    question_agent_model_name="gpt-5.4",
    max_loops=1,
    show_dashboard=True,
)

result = swarm.run(
    "What are the most promising near-term applications of small "
    "language models (under 10B parameters)?"
)

print(result)
