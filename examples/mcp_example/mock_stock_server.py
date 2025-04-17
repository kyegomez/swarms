
from fastmcp import FastMCP
import time

# Initialize MCP server
mcp = FastMCP("Stock-Mock-Server")

@mcp.tool()
def calculate_simple_moving_average(prices: list[float], period: int) -> float:
    """Calculate Simple Moving Average"""
    try:
        time.sleep(0.1)  # Simulate processing time
        if len(prices) < period:
            return {"error": "Not enough data points"}
        return sum(prices[-period:]) / period
    except Exception as e:
        return {"error": f"Error calculating SMA: {str(e)}"}

@mcp.tool()
def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values"""
    try:
        time.sleep(0.1)  # Simulate processing time
        return ((new_value - old_value) / old_value) * 100
    except Exception as e:
        return {"error": f"Error calculating percentage change: {str(e)}"}

if __name__ == "__main__":
    print("Starting Mock Stock Server on port 8001...")
    mcp.run(transport="sse", host="0.0.0.0", port=8001)
