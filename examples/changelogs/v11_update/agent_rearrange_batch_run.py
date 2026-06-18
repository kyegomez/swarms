from swarms import Agent, AgentRearrange

agents = [
    Agent(agent_name="Researcher", model_name="gpt-5.4"),
    Agent(agent_name="Writer", model_name="claude-sonnet-4-6"),
]

pipeline = AgentRearrange(
    agents=agents,
    flow="Researcher -> Writer",
)

# Now runs all 100 tasks concurrently with isolated agent state
results = pipeline.batch_run(
    tasks=[f"task_{i}" for i in range(1, 101)],
    batch_size=10,
)
print(results)
