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
    mcp_urls=[
        "http://0.0.0.0:8001/mcp",
        "http://0.0.0.0:8000/mcp",
    ],
    model_name="gpt-4o-mini",
    output_type="all",
)

# Create a markdown file with initial content
out = agent.run(
    # "Use the get_okx_crypto_price to get the price of solana  just put the name of the coin",
    "Use the create_agent tool that is specialized in creating agents"
)

print(out)
