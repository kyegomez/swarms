from swarms import Agent
from swarms.schemas.mcp_schemas import MCPConnection


mcp_config = MCPConnection(
    url="http://0.0.0.0:8000/mcp",
    # headers={"Authorization": "Bearer 1234567890"},
    timeout=5,
)


mcp_url = "http://0.0.0.0:8000/mcp"

# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    agent_description="Personal finance advisor agent",
    max_loops=1,
    mcp_url=mcp_url,
    output_type="all",
)

# Create a markdown file with initial content
out = agent.run(
    "Fetch the price for bitcoin on both functions get_htx_crypto_price and get_crypto_price",
)

print(out)
