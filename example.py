from swarms.structs.agent import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)

# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    agent_description="Personal finance advisor agent",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    max_loops=2,
    model_name="gpt-4o-mini",
    dynamic_temperature_enabled=True,
    interactive=False,
)

agent.run("Conduct an analysis of the best real undervalued ETFs")
