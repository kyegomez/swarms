
from fastmcp import FastMCP
from litellm import LiteLLM

# Initialize MCP server for math operations
mcp = FastMCP("Math-Server")

@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together"""
    return a + b

@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract b from a"""
    return a - b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together"""
    return a * b

@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b"""
    if b == 0:
        return {"error": "Cannot divide by zero"}
    return a / b

if __name__ == "__main__":
    print("Starting Math Server on port 6274...")
    llm = LiteLLM(model_name="gpt-4o-mini")
    mcp.run(transport="sse", host="0.0.0.0", port=6274)
