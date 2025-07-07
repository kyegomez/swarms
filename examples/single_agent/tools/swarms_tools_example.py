from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)
from swarms_tools import yahoo_finance_api

# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    agent_description="Personal finance advisor agent",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    max_loops=1,
    model_name="gpt-4o-mini",
    tools=[yahoo_finance_api],
    dynamic_temperature_enabled=True,
)

agent.run(
    "Fetch the data for nvidia and tesla both with the yahoo finance api"
)
