
from fastmcp import FastMCP
from typing import Dict, Any

# Initialize MCP server for math operations
mcp = FastMCP("Math-Server")

@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers"""
    return a + b

@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract b from a"""
    return a - b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    return a * b

@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b"""
    if b == 0:
        return {"error": "Cannot divide by zero"}
    return a / b

@mcp.tool()
def calculate_percentage(part: float, whole: float) -> float:
    """Calculate percentage"""
    if whole == 0:
        return {"error": "Cannot calculate percentage with zero total"}
    return (part / whole) * 100

if __name__ == "__main__":
    print("Starting Math Server on port 6274...")
    # Initialize LiteLLM with specific model
llm = LiteLLM(model_name="gpt-4o-mini")
mcp.run(transport="sse", host="0.0.0.0", port=6274)
