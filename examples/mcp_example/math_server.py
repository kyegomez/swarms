
from fastmcp import FastMCP
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Math-Server")

@mcp.tool(name="add")
def add(a: float, b: float) -> float:
    return float(a) + float(b)

@mcp.tool(name="subtract") 
def subtract(a: float, b: float) -> float:
    return float(a) - float(b)

@mcp.tool(name="multiply")
def multiply(a: float, b: float) -> float:
    return float(a) * float(b)

@mcp.tool(name="divide")
def divide(a: float, b: float) -> float:
    if float(b) == 0:
        raise ValueError("Cannot divide by zero")
    return float(a) / float(b)

if __name__ == "__main__":
    print("Starting Math Server on port 6274...")
    mcp.run(transport="sse", port=6274)
