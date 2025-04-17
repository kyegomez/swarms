
from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import FINANCIAL_AGENT_SYS_PROMPT
from swarms.tools.mcp_integration import MCPServerSseParams

# Configure MCP server connection
server = MCPServerSseParams(
    url="http://0.0.0.0:6274",
    headers={"Content-Type": "application/json"}
)

# Initialize agent with MCP capabilities
agent = Agent(
    agent_name="Math-Agent",
    agent_description="Agent that performs math operations",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    max_loops=1,
    mcp_servers=[server],
    streaming_on=True
)

# Test addition
result = agent.run("Use the add tool to add 5 and 3")
print("Addition result:", result)

# Test multiplication
result = agent.run("Use the multiply tool to multiply 4 and 6")
print("Multiplication result:", result)
