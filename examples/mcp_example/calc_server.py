
from fastmcp import FastMCP
from litellm import LiteLLM
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Initialize MCP server for financial calculations
mcp = FastMCP("Calc-Server")

@mcp.tool(name="compound_interest", description="Calculate compound interest")
def compound_interest(principal: float, rate: float, time: float) -> float:
    try:
        result = principal * (1 + rate/100) ** time
        return round(result, 2)
    except Exception as e:
        logger.error(f"Error calculating compound interest: {e}")
        raise

@mcp.tool(name="percentage", description="Calculate percentage")
def percentage(value: float, percent: float) -> float:
    try:
        return (value * percent) / 100
    except Exception as e:
        logger.error(f"Error calculating percentage: {e}")
        raise

if __name__ == "__main__":
    print("Starting Calculation Server on port 6275...")
    llm = LiteLLM(system_prompt="You are a financial calculation expert.")
    mcp.run(transport="sse", host="0.0.0.0", port=6275)
