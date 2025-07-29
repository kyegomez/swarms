from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)


# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    agent_description="Personal finance advisor agent",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    max_loops=1,
    mcp_url="http://0.0.0.0:8000/sse",
    model_name="gpt-4o-mini",
    output_type="all",
)

# Create a markdown file with initial content
out = agent.run(
    "Use the get_okx_crypto_volume to get the volume of BTC just put the name of the coin",
)

print(out)
print(type(out))
