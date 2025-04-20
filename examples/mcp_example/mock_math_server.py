
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
    # FastMCP expects transport_kwargs as separate parameters
    mcp.run(transport="sse", transport_kwargs={"host": "0.0.0.0", "port": 8000})
