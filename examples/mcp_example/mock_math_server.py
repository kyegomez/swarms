from fastmcp import FastMCP
from loguru import logger
import time

# Create the MCP server with all interfaces binding
mcp = FastMCP(
    host=
    "0.0.0.0",  # Bind to all interfaces to be accessible from other contexts
    port=8000,
    transport="sse",
    require_session_id=False,
    cors_allowed_origins=["*"],  # Allow all origins for testing
    debug=True  # Enable debug mode
)


# Define tools
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
    # Log server details
    logger.info("Starting math server on http://0.0.0.0:8000")
    print("Math MCP Server is running on http://0.0.0.0:8000")
    print("Press Ctrl+C to stop.")

    # List available tools
    print("\nAvailable tools:")
    print("- add: Add two numbers")
    print("- multiply: Multiply two numbers")
    print("- divide: Divide first number by second number")

    # Add a small delay to ensure logging is complete
    time.sleep(0.5)

    # Run the MCP server
    mcp.run()
  except KeyboardInterrupt:
    logger.info("Server shutdown requested")
    print("\nShutting down server...")
  except Exception as e:
    logger.error(f"Server error: {e}")
    raise
