
from fastmcp import FastMCP
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Calc-Server")

@mcp.tool(name="compound_interest")
def compound_interest(principal: float, rate: float, time: float) -> float:
    return principal * (1 + rate/100) ** time

@mcp.tool(name="simple_interest")
def simple_interest(principal: float, rate: float, time: float) -> float:
    return (principal * rate * time) / 100

if __name__ == "__main__":
    print("Starting Calculation Server on port 6275...")
    mcp.run(transport="sse", host="0.0.0.0", port=6275)
