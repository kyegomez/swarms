
from fastmcp import FastMCP
from typing import Dict, Union

# Create FastMCP server
mcp = FastMCP("Stock-Mock-Server")

@mcp.tool()
def get_stock_price(symbol: str) -> Dict[str, Union[float, str]]:
    """Get the current price of a stock"""
    prices = {
        "AAPL": 150.0,
        "GOOGL": 2800.0,
        "MSFT": 300.0,
        "AMZN": 3300.0
    }
    if symbol not in prices:
        return {"error": f"Stock {symbol} not found"}
    return {"price": prices[symbol]}

@mcp.tool()
def calculate_moving_average(prices: list[float], window: int) -> Dict[str, Union[list[float], str]]:
    """Calculate moving average of stock prices"""
    if not isinstance(prices, list) or not all(isinstance(x, (int, float)) for x in prices):
        return {"error": "Invalid price data"}
    if not isinstance(window, int) or window <= 0:
        return {"error": "Invalid window size"}
    if len(prices) < window:
        return {"error": "Not enough price points"}
    
    avgs = []
    for i in range(len(prices) - window + 1):
        avg = sum(prices[i:i+window]) / window
        avgs.append(round(avg, 2))
    return {"averages": avgs}

if __name__ == "__main__":
    print("Starting Mock Stock Server on port 8001...")
    mcp.run(transport="sse", transport_kwargs={"host": "0.0.0.0", "port": 8001})
