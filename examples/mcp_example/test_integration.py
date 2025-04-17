
from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import FINANCIAL_AGENT_SYS_PROMPT
from swarms.tools.mcp_integration import MCPServerSseParams
import logging
from typing import Dict, Any, Optional

def handle_mcp_response(response: Dict[str, Any]) -> str:
    """Handle MCP response and extract meaningful output"""
    if response.get("status") == "error":
        return f"Error: {response.get('message', 'Unknown error occurred')}"
    return str(response.get("result", response))

def setup_mcp_agent(name: str, description: str) -> Agent:
    """Setup an MCP-enabled agent with proper configuration"""
    try:
        server = MCPServerSseParams(
            url="http://0.0.0.0:6274",
            headers={"Content-Type": "application/json"},
            timeout=10.0,
            sse_read_timeout=300.0
        )
        
        return Agent(
            agent_name=name,
            agent_description=description,
            system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
            max_loops=1,
            mcp_servers=[server],
            streaming_on=True
        )
    except Exception as e:
        logging.error(f"Failed to setup agent: {e}")
        raise

def test_mcp_operations():
    """Test basic MCP operations with error handling"""
    try:
        # Initialize agent
        agent = setup_mcp_agent(
            "Math-Agent",
            "Agent that performs math operations"
        )

        # Get available operations
        print("\nQuerying available operations...")
        result = agent.run("What operations are available?")
        print("Available operations:", result)

        # Test addition
        print("\nTesting addition...")
        add_result = agent.run("Use the add tool to add 5 and 3")
        print("Addition result:", handle_mcp_response(add_result))

        # Test multiplication
        print("\nTesting multiplication...")
        mult_result = agent.run("Use the multiply tool to multiply 4 and 6")
        print("Multiplication result:", handle_mcp_response(mult_result))

        # Test error case
        print("\nTesting error handling...")
        error_result = agent.run("Use the add tool with invalid inputs")
        print("Error handling result:", handle_mcp_response(error_result))

    except Exception as e:
        logging.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_mcp_operations()
