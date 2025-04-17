from fastmcp import FastMCP
from typing import Dict, Any, Optional

# Initialize MCP server
mcp = FastMCP("Math-Server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    try:
        return a + b
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together"""
    try:
        return a * b 
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("Starting Math Server...")
    mcp.run(transport="sse")