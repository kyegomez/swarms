from swarms import HeavySwarm


swarm = HeavySwarm(
    name="ResearchSwarm",
    worker_model_name="gpt-5.4",
    question_agent_model_name="gpt-5.4",
    variant="medium",  # try "default" (5) or "heavy" (16)
    max_loops=1,
)

result = swarm.run(
    "What are the most promising near-term applications of small "
    "language models (under 10B parameters)?"
)

print(result)
