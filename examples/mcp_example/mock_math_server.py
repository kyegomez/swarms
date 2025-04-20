
from fastmcp import FastMCP
from loguru import logger

mcp = FastMCP(
    host="0.0.0.0",
    port=8000,
    transport="sse",
    require_session_id=False
)

@mcp.tool()
def add(a: int, b: int) -> str:
    """Add two numbers."""
    result = a + b
    return f"The sum of {a} and {b} is {result}"

@mcp.tool() 
def multiply(a: int, b: int) -> str:
    """Multiply two numbers."""
    result = a * b
    return f"The product of {a} and {b} is {result}"

@mcp.tool()
def divide(a: int, b: int) -> str:
    """Divide two numbers."""
    if b == 0:
        return "Cannot divide by zero"
    result = a / b
    return f"{a} divided by {b} is {result}"

if __name__ == "__main__":
    try:
        logger.info("Starting math server on http://0.0.0.0:8000")
        mcp.run()
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
