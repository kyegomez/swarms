
from fastmcp import FastMCP
from typing import Dict, Any
import math

# Initialize MCP server
mcp = FastMCP("Calc-Server")

@mcp.tool()
def square_root(x: float) -> float:
    """Calculate square root of a number"""
    try:
        return math.sqrt(x)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def power(base: float, exponent: float) -> float:
    """Raise a number to a power"""
    try:
        return math.pow(base, exponent)
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("Starting Calc Server on port 6275...")
    mcp.run(transport="sse", host="0.0.0.0", port=6275)
