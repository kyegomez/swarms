import os

from swarms import Agent

# TeamoRouter serves GLM, Kimi, Qwen, Claude, GPT and other models behind one
# OpenAI-compatible endpoint. Needs TEAMOROUTER_API_KEY.
agent = Agent(
    agent_name="GLM-Agent",
    agent_description="Low-cost agent on GLM 5.3",
    model_name="openai/glm-5.3",
    llm_base_url="https://api.teamorouter.com/v1",
    llm_api_key=os.getenv("TEAMOROUTER_API_KEY"),
    max_loops=1,
)

out = agent.run(
    task="Compare SQLite and Postgres for a small web app in five bullet points.",
)

print(out)
