
from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)
from swarms.tools.mcp_integration import MCPServerSseParams

server_one = MCPServerSseParams(
    url="http://127.0.0.1:6274",
    headers={"Content-Type": "application/json"},
)

# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    agent_description="Personal finance advisor agent",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    max_loops=1,
    mcp_servers=[server_one],
    output_type="final",
)

out = agent.run("Use the add tool to add 2 and 2")

print(type(out))
