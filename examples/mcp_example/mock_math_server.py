from fastmcp import FastMCP
from loguru import logger
import sys
import time

# Configure detailed logging
logger.remove()
logger.add(sys.stdout, level="DEBUG", format="{time} | {level} | {message}")

# Create MCP server with fixed configuration
mcp = FastMCP(
    host="127.0.0.1",  # Bind to localhost only
    port=8000,
    transport="sse",
    require_session_id=False,
    cors_allowed_origins=["*"],
    debug=True
)

# Define tools with proper return format
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    result = a + b
    logger.info(f"Adding {a} + {b} = {result}")
    return result  # Let FastMCP handle the response formatting

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    result = a * b
    logger.info(f"Multiplying {a} * {b} = {result}")
    return result

@mcp.tool()
def divide(a: int, b: int) -> float:
    """Divide the first number by the second."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    result = a / b
    logger.info(f"Dividing {a} / {b} = {result}")
    return result

def main():
    try:
        logger.info("Starting mock math server on http://127.0.0.1:8000")
        print("Math MCP Server running on http://127.0.0.1:8000 (SSE)\n")
        print("Available tools:\n - add\n - multiply\n - divide\n")
        mcp.run()  # This runs the server in a blocking mode
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()