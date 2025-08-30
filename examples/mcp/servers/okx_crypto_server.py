from mcp.server.fastmcp import FastMCP
import requests

mcp = FastMCP("OKXCryptoPrice")

mcp.settings.port = 8001


@mcp.tool(
    name="get_okx_crypto_price",
    description="Get the current price and basic information for a given cryptocurrency from OKX exchange.",
)
def get_okx_crypto_price(symbol: str) -> str:
    """
    Get the current price and basic information for a given cryptocurrency using OKX API.

    Args:
        symbol (str): The cryptocurrency trading pair (e.g., 'BTC-USDT', 'ETH-USDT')

    Returns:
        str: A formatted string containing the cryptocurrency information

    Example:
        >>> get_okx_crypto_price('BTC-USDT')
        'Current price of BTC/USDT: $45,000'
    """
    try:
        if not symbol:
            return "Please provide a valid trading pair (e.g., 'BTC-USDT')"

        # Convert to uppercase and ensure proper format
        symbol = symbol.upper()
        if not symbol.endswith("-USDT"):
            symbol = f"{symbol}-USDT"

        # OKX API endpoint for ticker information
        url = f"https://www.okx.com/api/v5/market/ticker?instId={symbol}"

        # Make the API request
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()

        if data.get("code") != "0":
            return f"Error: {data.get('msg', 'Unknown error')}"

        ticker_data = data.get("data", [{}])[0]
        if not ticker_data:
            return f"Could not find data for {symbol}. Please check the trading pair."

        price = float(ticker_data.get("last", 0))
        float(ticker_data.get("last24h", 0))
        change_percent = float(ticker_data.get("change24h", 0))

        base_currency = symbol.split("-")[0]
        return f"Current price of {base_currency}/USDT: ${price:,.2f}\n24h Change: {change_percent:.2f}%"

    except requests.exceptions.RequestException as e:
        return f"Error fetching OKX data: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool(
    name="get_okx_crypto_volume",
    description="Get the 24-hour trading volume for a given cryptocurrency from OKX exchange.",
)
def get_okx_crypto_volume(symbol: str) -> str:
    """
    Get the 24-hour trading volume for a given cryptocurrency using OKX API.

    Args:
        symbol (str): The cryptocurrency trading pair (e.g., 'BTC-USDT', 'ETH-USDT')

    Returns:
        str: A formatted string containing the trading volume information

    Example:
        >>> get_okx_crypto_volume('BTC-USDT')
        '24h Trading Volume for BTC/USDT: $1,234,567'
    """
    try:
        if not symbol:
            return "Please provide a valid trading pair (e.g., 'BTC-USDT')"

        # Convert to uppercase and ensure proper format
        symbol = symbol.upper()
        if not symbol.endswith("-USDT"):
            symbol = f"{symbol}-USDT"

        # OKX API endpoint for ticker information
        url = f"https://www.okx.com/api/v5/market/ticker?instId={symbol}"

        # Make the API request
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()

        if data.get("code") != "0":
            return f"Error: {data.get('msg', 'Unknown error')}"

        ticker_data = data.get("data", [{}])[0]
        if not ticker_data:
            return f"Could not find data for {symbol}. Please check the trading pair."

        volume_24h = float(ticker_data.get("vol24h", 0))
        base_currency = symbol.split("-")[0]
        return f"24h Trading Volume for {base_currency}/USDT: ${volume_24h:,.2f}"

    except requests.exceptions.RequestException as e:
        return f"Error fetching OKX data: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    # Run the server on port 8000 (you can change this to any available port)
    mcp.run(transport="sse")
