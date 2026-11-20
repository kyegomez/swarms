import os

from swarms import Agent

# Same endpoint and key as glm_teamorouter.py; only the model id changes.
agent = Agent(
    agent_name="Kimi-Agent",
    agent_description="Long-context agent on Kimi K3.1 Pro",
    model_name="openai/kimi-k3.1-pro",
    llm_base_url="https://api.teamorouter.com/v1",
    llm_api_key=os.getenv("TEAMOROUTER_API_KEY"),
    max_loops=1,
)

out = agent.run(
    task="Summarize the trade-offs of optimistic and pessimistic locking in one paragraph.",
)

print(out)
