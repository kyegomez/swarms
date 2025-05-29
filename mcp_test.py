# crypto_price_server.py
from mcp.server.fastmcp import FastMCP
import requests

mcp = FastMCP("CryptoPrice")


@mcp.tool(
    name="get_crypto_price",
    description="Get the current price and basic information for a given cryptocurrency.",
)
def get_crypto_price(coin_id: str) -> str:
    """
    Get the current price and basic information for a given cryptocurrency using CoinGecko API.

    Args:
        coin_id (str): The cryptocurrency ID (e.g., 'bitcoin', 'ethereum')

    Returns:
        str: A formatted string containing the cryptocurrency information

    Example:
        >>> get_crypto_price('bitcoin')
        'Current price of Bitcoin: $45,000'
    """
    try:
        if not coin_id:
            return "Please provide a valid cryptocurrency ID"

        # CoinGecko API endpoint
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_change=true"

        # Make the API request
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        data = response.json()

        if coin_id not in data:
            return f"Could not find data for {coin_id}. Please check the cryptocurrency ID."

        price = data[coin_id]["usd"]
        change_24h = data[coin_id].get("usd_24h_change", "N/A")

        return f"Current price of {coin_id.capitalize()}: ${price:,.2f}\n24h Change: {change_24h:.2f}%"

    except requests.exceptions.RequestException as e:
        return f"Error fetching crypto data: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool(
    name="get_htx_crypto_price",
    description="Get the current price and basic information for a given cryptocurrency from HTX exchange.",
)
def get_htx_crypto_price(symbol: str) -> str:
    """
    Get the current price and basic information for a given cryptocurrency using HTX API.

    Args:
        symbol (str): The cryptocurrency trading pair (e.g., 'btcusdt', 'ethusdt')

    Returns:
        str: A formatted string containing the cryptocurrency information

    Example:
        >>> get_htx_crypto_price('btcusdt')
        'Current price of BTC/USDT: $45,000'
    """
    try:
        if not symbol:
            return "Please provide a valid trading pair (e.g., 'btcusdt')"

        # Convert to lowercase and ensure proper format
        symbol = symbol.lower()
        if not symbol.endswith("usdt"):
            symbol = f"{symbol}usdt"

        # HTX API endpoint
        url = f"https://api.htx.com/market/detail/merged?symbol={symbol}"

        # Make the API request
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()

        if data.get("status") != "ok":
            return f"Error: {data.get('err-msg', 'Unknown error')}"

        tick = data.get("tick", {})
        if not tick:
            return f"Could not find data for {symbol}. Please check the trading pair."

        price = tick.get("close", 0)
        change_24h = tick.get("close", 0) - tick.get("open", 0)
        change_percent = (
            (change_24h / tick.get("open", 1)) * 100
            if tick.get("open")
            else 0
        )

        base_currency = symbol[
            :-4
        ].upper()  # Remove 'usdt' and convert to uppercase
        return f"Current price of {base_currency}/USDT: ${price:,.2f}\n24h Change: {change_percent:.2f}%"

    except requests.exceptions.RequestException as e:
        return f"Error fetching HTX data: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="sse")
