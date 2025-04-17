import logging
from fastmcp import FastMCP
from litellm import LiteLLM

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Initialize MCP server for math operations
mcp = FastMCP("Math-Server")

@mcp.tool(name="add", description="Add two numbers") 
def add(a: float, b: float) -> float:
    try:
        result = float(a) + float(b)
        return result
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid input types for addition: {e}")
        raise ValueError("Inputs must be valid numbers")
    except Exception as e:
        logger.error(f"Unexpected error in add operation: {e}")
        raise

@mcp.tool(name="subtract", description="Subtract b from a")
def subtract(a: float, b: float) -> float:
    try:
        result = float(a) - float(b)
        return result
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid input types for subtraction: {e}")
        raise ValueError("Inputs must be valid numbers")
    except Exception as e:
        logger.error(f"Unexpected error in subtract operation: {e}")
        raise

@mcp.tool(name="multiply", description="Multiply two numbers together")
def multiply(a: float, b: float) -> float:
    try:
        result = float(a) * float(b)
        return result
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid input types for multiplication: {e}")
        raise ValueError("Inputs must be valid numbers")
    except Exception as e:
        logger.error(f"Unexpected error in multiply operation: {e}")
        raise

@mcp.tool(name="divide", description="Divide a by b")
def divide(a: float, b: float) -> float:
    try:
        if float(b) == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        result = float(a) / float(b)
        return result
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid input types for division: {e}")
        raise ValueError("Inputs must be valid numbers")
    except ZeroDivisionError as e:
        logger.error(f"ZeroDivisionError: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in divide operation: {e}")
        raise



if __name__ == "__main__":
    print("Starting Math Server on port 6274...")
    llm = LiteLLM(model_name="gpt-4o-mini")
    mcp.run(transport="sse", host="0.0.0.0", port=6274)