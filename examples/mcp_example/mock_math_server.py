from fastmcp import FastMCP
from loguru import logger

mcp = FastMCP(
    host="0.0.0.0",
    port=8000,
    transport="sse",
    require_session_id=False,
    timeout=30.0
)

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@mcp.tool() 
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@mcp.tool()
def divide(a: int, b: int) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

if __name__ == "__main__":
    try:
        logger.info("Starting math server on http://0.0.0.0:8000")
        mcp.run()
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise