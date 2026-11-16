from swarms import Agent

# Needs OPENROUTER_API_KEY.
agent = Agent(
    agent_name="GLM-Flash-Agent",
    agent_description="Fast, low-cost agent on GLM 5.3 Flash",
    model_name="openrouter/z-ai/glm-5.3-flash",
    max_loops=1,
    persistent_memory=False,
)

out = agent.run(
    task="Compare SQLite and Postgres for a small web app in five bullet points.",
)

print(out)
