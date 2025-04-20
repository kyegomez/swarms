
from fastmcp import FastMCP

mcp = FastMCP("Math-Mock-Server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together"""
    return a * b

@mcp.tool()
def divide(a: int, b: int) -> float:
    """Divide two numbers"""
    if b == 0:
        return {"error": "Cannot divide by zero"}
    return a / b

if __name__ == "__main__":
    print("Starting Mock Math Server on port 8000...")
    # Fix the parameters to match the FastMCP API
    mcp.run(transport="sse", port=8000)
