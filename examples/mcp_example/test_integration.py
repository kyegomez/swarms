from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import FINANCIAL_AGENT_SYS_PROMPT
from swarms.tools.mcp_integration import MCPServerSseParams
import logging

# Configure MCP server connection
server = MCPServerSseParams(
    url="http://0.0.0.0:6274",
    headers={"Content-Type": "application/json"},
    timeout=10.0,
    sse_read_timeout=300.0
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

def test_mcp_operations():
    """Test basic MCP operations with error handling"""
    try:
        # Test addition
        print("\nTesting addition...")
        add_result = agent.run("Use the add tool to add 5 and 3")
        print("Addition result:", add_result)

        # Test multiplication
        print("\nTesting multiplication...")
        mult_result = agent.run("Use the multiply tool to multiply 4 and 6") 
        print("Multiplication result:", mult_result)

        # Test error case
        print("\nTesting error handling...")
        error_result = agent.run("Use the add tool with invalid inputs")
        print("Error handling result:", error_result)

    except Exception as e:
        logging.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_mcp_operations()