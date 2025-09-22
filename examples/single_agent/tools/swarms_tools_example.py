from swarms import Agent
from swarms_tools.finance.okx_tool import okx_api_tool

# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    agent_description="Personal finance advisor agent",
    max_loops=1,
    model_name="gpt-4o-mini",
    tools=[okx_api_tool],
    dynamic_temperature_enabled=True,
)

agent.run("fetch the current price of bitcoin with okx")
