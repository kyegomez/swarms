
from fastmcp import FastMCP
from typing import Dict, Any
import time

# Initialize MCP server
mcp = FastMCP("Math-Mock-Server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    time.sleep(0.1)  # Simulate processing time
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together"""
    time.sleep(0.1)  # Simulate processing time
    return a * b

@mcp.tool()
def divide(a: int, b: int) -> float:
    """Divide two numbers"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    time.sleep(0.1)  # Simulate processing time
    return a / b

if __name__ == "__main__":
    print("Starting Mock Math Server on port 8000...")
    mcp.run(transport="sse", port=8000, host="0.0.0.0")
