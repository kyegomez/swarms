from fastmcp import FastMCP
from loguru import logger
import time

# Create the MCP server
mcp = FastMCP(host="0.0.0.0",
              port=8000,
              transport="sse",
              require_session_id=False)


# Define tools with proper type hints and docstrings
@mcp.tool()
def add(a: int, b: int) -> str:
    """Add two numbers.

    Args:
        a (int): First number
        b (int): Second number

    Returns:
        str: A message containing the sum
    """
    logger.info(f"Adding {a} and {b}")
    result = a + b
    return f"The sum of {a} and {b} is {result}"


@mcp.tool()
def multiply(a: int, b: int) -> str:
    """Multiply two numbers.

    Args:
        a (int): First number
        b (int): Second number

    Returns:
        str: A message containing the product
    """
    logger.info(f"Multiplying {a} and {b}")
    result = a * b
    return f"The product of {a} and {b} is {result}"


@mcp.tool()
def divide(a: int, b: int) -> str:
    """Divide two numbers.

    Args:
        a (int): Numerator
        b (int): Denominator

    Returns:
        str: A message containing the division result or an error message
    """
    logger.info(f"Dividing {a} by {b}")
    if b == 0:
        logger.warning("Division by zero attempted")
        return "Cannot divide by zero"
    result = a / b
    return f"{a} divided by {b} is {result}"


if __name__ == "__main__":
    try:
        logger.info("Starting math server on http://0.0.0.0:8000")
        print("Math MCP Server is running. Press Ctrl+C to stop.")

        # Add a small delay to ensure logging is complete before the server starts
        time.sleep(0.5)

        # Run the MCP server
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
        print("\nShutting down server...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
