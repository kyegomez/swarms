from fastmcp import FastMCP
from loguru import logger
import time
import json

# Create the MCP server with detailed debugging
mcp = FastMCP(
    host="0.0.0.0",  # Bind to all interfaces
    port=8000,
    transport="sse",
    require_session_id=False,
    cors_allowed_origins=["*"],  # Allow connections from any origin
    debug=True  # Enable debug mode for more verbose output
)


# Add a more flexible parsing approach
def parse_input(input_str):
    """Parse input that could be JSON or natural language."""
    try:
        # First try to parse as JSON
        return json.loads(input_str)
    except json.JSONDecodeError:
        # If not JSON, try to parse natural language
        input_lower = input_str.lower()

        # Parse for addition
        if "add" in input_lower or "plus" in input_lower or "sum" in input_lower:
            # Extract numbers - very simple approach
            numbers = [int(s) for s in input_lower.split() if s.isdigit()]
            if len(numbers) >= 2:
                return {"a": numbers[0], "b": numbers[1]}

        # Parse for multiplication
        if "multiply" in input_lower or "times" in input_lower or "product" in input_lower:
            numbers = [int(s) for s in input_lower.split() if s.isdigit()]
            if len(numbers) >= 2:
                return {"a": numbers[0], "b": numbers[1]}

        # Parse for division
        if "divide" in input_lower or "quotient" in input_lower:
            numbers = [int(s) for s in input_lower.split() if s.isdigit()]
            if len(numbers) >= 2:
                return {"a": numbers[0], "b": numbers[1]}

        # Could not parse successfully
        return None


# Define tools with more flexible input handling
@mcp.tool()
def add(input_str=None, a=None, b=None):
    """Add two numbers. Can accept JSON parameters or natural language.

    Args:
        input_str (str, optional): Natural language input to parse
        a (int, optional): First number if provided directly
        b (int, optional): Second number if provided directly

    Returns:
        str: A message containing the sum
    """
    logger.info(f"Add tool called with input_str={input_str}, a={a}, b={b}")

    # If we got a natural language string instead of parameters
    if input_str and not (a is not None and b is not None):
        parsed = parse_input(input_str)
        if parsed:
            a = parsed.get("a")
            b = parsed.get("b")

    # Validate we have what we need
    if a is None or b is None:
        return "Sorry, I couldn't understand the numbers to add"

    try:
        a = int(a)
        b = int(b)
        result = a + b
        return f"The sum of {a} and {b} is {result}"
    except ValueError:
        return "Please provide valid numbers for addition"


@mcp.tool()
def multiply(input_str=None, a=None, b=None):
    """Multiply two numbers. Can accept JSON parameters or natural language.

    Args:
        input_str (str, optional): Natural language input to parse
        a (int, optional): First number if provided directly
        b (int, optional): Second number if provided directly

    Returns:
        str: A message containing the product
    """
    logger.info(
        f"Multiply tool called with input_str={input_str}, a={a}, b={b}")

    # If we got a natural language string instead of parameters
    if input_str and not (a is not None and b is not None):
        parsed = parse_input(input_str)
        if parsed:
            a = parsed.get("a")
            b = parsed.get("b")

    # Validate we have what we need
    if a is None or b is None:
        return "Sorry, I couldn't understand the numbers to multiply"

    try:
        a = int(a)
        b = int(b)
        result = a * b
        return f"The product of {a} and {b} is {result}"
    except ValueError:
        return "Please provide valid numbers for multiplication"


@mcp.tool()
def divide(input_str=None, a=None, b=None):
    """Divide two numbers. Can accept JSON parameters or natural language.

    Args:
        input_str (str, optional): Natural language input to parse
        a (int, optional): Numerator if provided directly
        b (int, optional): Denominator if provided directly

    Returns:
        str: A message containing the division result or an error message
    """
    logger.info(f"Divide tool called with input_str={input_str}, a={a}, b={b}")

    # If we got a natural language string instead of parameters
    if input_str and not (a is not None and b is not None):
        parsed = parse_input(input_str)
        if parsed:
            a = parsed.get("a")
            b = parsed.get("b")

    # Validate we have what we need
    if a is None or b is None:
        return "Sorry, I couldn't understand the numbers to divide"

    try:
        a = int(a)
        b = int(b)

        if b == 0:
            logger.warning("Division by zero attempted")
            return "Cannot divide by zero"

        result = a / b
        return f"{a} divided by {b} is {result}"
    except ValueError:
        return "Please provide valid numbers for division"


if __name__ == "__main__":
    try:
        logger.info("Starting math server on http://0.0.0.0:8000")
        print("Math MCP Server is running. Press Ctrl+C to stop.")
        print(
            "Server is configured to accept both JSON and natural language input"
        )

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
