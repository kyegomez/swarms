
from fastmcp import FastMCP

mcp = FastMCP("Math-Server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    mcp.run(host="0.0.0.0", port=6274, transport="sse")
