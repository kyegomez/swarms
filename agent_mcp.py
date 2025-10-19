from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)

agent = Agent(
    agent_name="Financial-Analysis-Agent",  # Name of the agent
    agent_description="Personal finance advisor agent",  # Description of the agent's role
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,  # System prompt for financial tasks
    max_loops=1,
    mcp_urls=[
        "http://0.0.0.0:5932/mcp",
    ],
    model_name="gpt-4o-mini",
    output_type="all",
)

out = agent.run(
    "Use the discover agent tools to find what agents are available and provide a summary"
)

# Print the output from the agent's run method.
print(out)
