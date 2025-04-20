from fastmcp import FastMCP
from typing import Dict, Any
import asyncio
from loguru import logger

# Create FastMCP instance with SSE transport
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

async def run_server():
    """Run the server with proper error handling."""
    try:
        logger.info("Starting math server on http://0.0.0.0:8000")
        await mcp.run_async()
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        await mcp.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
