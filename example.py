from swarms import Agent
from swarms_tools import exa_search

# Initialize the agent
agent = Agent(
    agent_name="Quantitative-Trading-Agent",
    agent_description="Advanced quantitative trading and algorithmic analysis agent",
    model_name="claude-sonnet-4-20250514",
    dynamic_temperature_enabled=True,
    max_loops=1,
    dynamic_context_window=True,
    tools=[exa_search],
    streaming_on=False,
)

out = agent.run(
    task="What are the best top 3 etfs for gold coverage?"
)

print(out)
