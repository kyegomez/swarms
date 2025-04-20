
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
    # The port parameter should be included in the transport_kwargs dictionary
    mcp.run(transport="sse", transport_kwargs={"port": 8000})
