
from fastmcp import FastMCP
from typing import Dict, Any
import time

# Initialize MCP server
mcp = FastMCP("Math-Mock-Server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    try:
        print(f"MCP Server: Processing addition request for {a} + {b}")
        time.sleep(0.1)  # Simulate processing time
        result = a + b
        print(f"MCP Server: Returning result {result}")
        return result
    except Exception as e:
        error_msg = f"Error adding numbers: {str(e)}"
        print(f"MCP Server: {error_msg}")
        return {"error": error_msg}

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together"""
    try:
        time.sleep(0.1)  # Simulate processing time
        return a * b
    except Exception as e:
        return {"error": f"Error multiplying numbers: {str(e)}"}

@mcp.tool()
def divide(a: int, b: int) -> float:
    """Divide two numbers"""
    try:
        if b == 0:
            return {"error": "Cannot divide by zero"}
        time.sleep(0.1)  # Simulate processing time
        return a / b
    except Exception as e:
        return {"error": f"Error dividing numbers: {str(e)}"}

if __name__ == "__main__":
    print("Starting Mock Math Server on port 8000...")
    # Initialize connection params
    server_params = MCPServerSseParams(
        url="http://0.0.0.0:8000/mcp",
        headers={"Content-Type": "application/json"},
        timeout=5.0,
        sse_read_timeout=30.0
    )
    
    # Initialize the agent
    agent = Agent(
        agent_name="Math-Processing-Agent",
        agent_description="Specialized agent for mathematical operations",
        system_prompt="You are a mathematical computation specialist. Use the available tools to perform calculations.",
        max_loops=1,
        mcp_servers=[server_params],
        streaming_on=True,
        model_name="gpt-4o-mini"
    )
    
    # Start the MCP server
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
