
from fastmcp import Server
from fastmcp.messages import Request, Response

# Create MCP server
mcp = Server("Stock-Mock-Server")

# Define stock price lookup handler
@mcp.handler("get_stock_price")
async def get_stock_price(request: Request) -> Response:
    # Mock stock price data
    stock_prices = {
        "AAPL": 150.0,
        "GOOGL": 2800.0,
        "MSFT": 300.0
    }
    
    symbol = request.data.get("symbol")
    if symbol in stock_prices:
        return Response(data={"price": stock_prices[symbol]})
    else:
        return Response(error=f"Stock {symbol} not found")

if __name__ == "__main__":
    print("Starting Mock Stock Server on port 8001...")
    # Run server with SSE transport on specified host/port
    mcp.run(
        transport="sse",
        transport_kwargs={
            "host": "0.0.0.0",
            "port": 8001
        }
    )
