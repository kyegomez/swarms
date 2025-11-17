from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)

agent = Agent(
    agent_name="Financial-Analysis-Agent",  # Name of the agent
    agent_description="Personal finance advisor agent",  # Description of the agent's role
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,  # System prompt for financial tasks
    max_loops=1,
    mcp_url="http://0.0.0.0:8001/mcp",  # URL for the OKX crypto price MCP server
    model_name="gpt-4o-mini",
    output_type="all",
)

out = agent.run(
    "Use the get_okx_crypto_price to get the price of solana  just put the name of the coin",
)

# Print the output from the agent's run method.
print(out)
