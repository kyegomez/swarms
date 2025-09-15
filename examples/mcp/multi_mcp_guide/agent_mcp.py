from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)

# Initialize the financial analysis agent with a system prompt and configuration.
agent = Agent(
    agent_name="Financial-Analysis-Agent",  # Name of the agent
    agent_description="Personal finance advisor agent",  # Description of the agent's role
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,  # System prompt for financial tasks
    max_loops=1,
    mcp_urls=[
        "http://0.0.0.0:8001/mcp",  # URL for the OKX crypto price MCP server
        "http://0.0.0.0:8000/mcp",  # URL for the agent creation MCP server
    ],
    model_name="gpt-4o-mini",
    output_type="all",
)

# Run the agent with a specific instruction to use the create_agent tool.
# The agent is asked to create a new agent specialized for accounting rules in crypto.
out = agent.run(
    # Example alternative prompt:
    # "Use the get_okx_crypto_price to get the price of solana  just put the name of the coin",
    "Use the create_agent tool that is specialized in creating agents and create an agent speecialized for accounting rules in crypto"
)

# Print the output from the agent's run method.
print(out)
