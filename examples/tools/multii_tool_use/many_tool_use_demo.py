import json
import requests
from swarms import Agent
from typing import List
import time


def get_coin_price(coin_id: str, vs_currency: str) -> str:
    """
    Get the current price of a specific cryptocurrency.

    Args:
        coin_id (str): The CoinGecko ID of the cryptocurrency (e.g., 'bitcoin', 'ethereum')
        vs_currency (str, optional): The target currency. Defaults to "usd".

    Returns:
        str: JSON formatted string containing the coin's current price and market data

    Raises:
        requests.RequestException: If the API request fails

    Example:
        >>> result = get_coin_price("bitcoin")
        >>> print(result)
        {"bitcoin": {"usd": 45000, "usd_market_cap": 850000000000, ...}}
    """
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": coin_id,
            "vs_currencies": vs_currency,
            "include_market_cap": True,
            "include_24hr_vol": True,
            "include_24hr_change": True,
            "include_last_updated_at": True,
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        return json.dumps(data, indent=2)

    except requests.RequestException as e:
        return json.dumps(
            {
                "error": f"Failed to fetch price for {coin_id}: {str(e)}"
            }
        )
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


def get_top_cryptocurrencies(limit: int, vs_currency: str) -> str:
    """
    Fetch the top cryptocurrencies by market capitalization.

    Args:
        limit (int, optional): Number of coins to retrieve (1-250). Defaults to 10.
        vs_currency (str, optional): The target currency. Defaults to "usd".

    Returns:
        str: JSON formatted string containing top cryptocurrencies with detailed market data

    Raises:
        requests.RequestException: If the API request fails
        ValueError: If limit is not between 1 and 250

    Example:
        >>> result = get_top_cryptocurrencies(5)
        >>> print(result)
        [{"id": "bitcoin", "name": "Bitcoin", "current_price": 45000, ...}]
    """
    try:
        if not 1 <= limit <= 250:
            raise ValueError("Limit must be between 1 and 250")

        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1,
            "sparkline": False,
            "price_change_percentage": "24h,7d",
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Simplify the data structure for better readability
        simplified_data = []
        for coin in data:
            simplified_data.append(
                {
                    "id": coin.get("id"),
                    "symbol": coin.get("symbol"),
                    "name": coin.get("name"),
                    "current_price": coin.get("current_price"),
                    "market_cap": coin.get("market_cap"),
                    "market_cap_rank": coin.get("market_cap_rank"),
                    "total_volume": coin.get("total_volume"),
                    "price_change_24h": coin.get(
                        "price_change_percentage_24h"
                    ),
                    "price_change_7d": coin.get(
                        "price_change_percentage_7d_in_currency"
                    ),
                    "last_updated": coin.get("last_updated"),
                }
            )

        return json.dumps(simplified_data, indent=2)

    except (requests.RequestException, ValueError) as e:
        return json.dumps(
            {
                "error": f"Failed to fetch top cryptocurrencies: {str(e)}"
            }
        )
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


def search_cryptocurrencies(query: str) -> str:
    """
    Search for cryptocurrencies by name or symbol.

    Args:
        query (str): The search term (coin name or symbol)

    Returns:
        str: JSON formatted string containing search results with coin details

    Raises:
        requests.RequestException: If the API request fails

    Example:
        >>> result = search_cryptocurrencies("ethereum")
        >>> print(result)
        {"coins": [{"id": "ethereum", "name": "Ethereum", "symbol": "eth", ...}]}
    """
    try:
        url = "https://api.coingecko.com/api/v3/search"
        params = {"query": query}

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Extract and format the results
        result = {
            "coins": data.get("coins", [])[
                :10
            ],  # Limit to top 10 results
            "query": query,
            "total_results": len(data.get("coins", [])),
        }

        return json.dumps(result, indent=2)

    except requests.RequestException as e:
        return json.dumps(
            {"error": f'Failed to search for "{query}": {str(e)}'}
        )
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


def get_jupiter_quote(
    input_mint: str,
    output_mint: str,
    amount: float,
    slippage: float = 0.5,
) -> str:
    """
    Get a quote for token swaps using Jupiter Protocol on Solana.

    Args:
        input_mint (str): Input token mint address
        output_mint (str): Output token mint address
        amount (float): Amount of input tokens to swap
        slippage (float, optional): Slippage tolerance percentage. Defaults to 0.5.

    Returns:
        str: JSON formatted string containing the swap quote details

    Example:
        >>> result = get_jupiter_quote("SOL_MINT_ADDRESS", "USDC_MINT_ADDRESS", 1.0)
        >>> print(result)
        {"inputAmount": "1000000000", "outputAmount": "22.5", "route": [...]}
    """
    try:
        url = "https://lite-api.jup.ag/swap/v1/quote"
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(int(amount * 1e9)),  # Convert to lamports
            "slippageBps": int(slippage * 100),
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)

    except requests.RequestException as e:
        return json.dumps(
            {"error": f"Failed to get Jupiter quote: {str(e)}"}
        )
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


def get_htx_market_data(symbol: str) -> str:
    """
    Get market data for a trading pair from HTX exchange.

    Args:
        symbol (str): Trading pair symbol (e.g., 'btcusdt', 'ethusdt')

    Returns:
        str: JSON formatted string containing market data

    Example:
        >>> result = get_htx_market_data("btcusdt")
        >>> print(result)
        {"symbol": "btcusdt", "price": "45000", "volume": "1000000", ...}
    """
    try:
        url = "https://api.htx.com/market/detail/merged"
        params = {"symbol": symbol.lower()}

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)

    except requests.RequestException as e:
        return json.dumps(
            {"error": f"Failed to fetch HTX market data: {str(e)}"}
        )
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


def get_token_historical_data(
    token_id: str, days: int = 30, vs_currency: str = "usd"
) -> str:
    """
    Get historical price and market data for a cryptocurrency.

    Args:
        token_id (str): The CoinGecko ID of the cryptocurrency
        days (int, optional): Number of days of historical data. Defaults to 30.
        vs_currency (str, optional): The target currency. Defaults to "usd".

    Returns:
        str: JSON formatted string containing historical price and market data

    Example:
        >>> result = get_token_historical_data("bitcoin", 7)
        >>> print(result)
        {"prices": [[timestamp, price], ...], "market_caps": [...], "volumes": [...]}
    """
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{token_id}/market_chart"
        params = {
            "vs_currency": vs_currency,
            "days": days,
            "interval": "daily",
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)

    except requests.RequestException as e:
        return json.dumps(
            {"error": f"Failed to fetch historical data: {str(e)}"}
        )
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


def get_defi_stats() -> str:
    """
    Get global DeFi statistics including TVL, trading volumes, and dominance.

    Returns:
        str: JSON formatted string containing global DeFi statistics

    Example:
        >>> result = get_defi_stats()
        >>> print(result)
        {"total_value_locked": 50000000000, "defi_dominance": 15.5, ...}
    """
    try:
        url = "https://api.coingecko.com/api/v3/global/decentralized_finance_defi"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)

    except requests.RequestException as e:
        return json.dumps(
            {"error": f"Failed to fetch DeFi stats: {str(e)}"}
        )
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


def get_jupiter_tokens() -> str:
    """
    Get list of tokens supported by Jupiter Protocol on Solana.

    Returns:
        str: JSON formatted string containing supported tokens

    Example:
        >>> result = get_jupiter_tokens()
        >>> print(result)
        {"tokens": [{"symbol": "SOL", "mint": "...", "decimals": 9}, ...]}
    """
    try:
        url = "https://lite-api.jup.ag/tokens/v1/mints/tradable"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)

    except requests.RequestException as e:
        return json.dumps(
            {"error": f"Failed to fetch Jupiter tokens: {str(e)}"}
        )
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


def get_htx_trading_pairs() -> str:
    """
    Get list of all trading pairs available on HTX exchange.

    Returns:
        str: JSON formatted string containing trading pairs information

    Example:
        >>> result = get_htx_trading_pairs()
        >>> print(result)
        {"symbols": [{"symbol": "btcusdt", "state": "online", "type": "spot"}, ...]}
    """
    try:
        url = "https://api.htx.com/v1/common/symbols"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)

    except requests.RequestException as e:
        return json.dumps(
            {"error": f"Failed to fetch HTX trading pairs: {str(e)}"}
        )
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


def get_market_sentiment(coin_ids: List[str]) -> str:
    """
    Get market sentiment data including social metrics and developer activity.

    Args:
        coin_ids (List[str]): List of CoinGecko coin IDs

    Returns:
        str: JSON formatted string containing market sentiment data

    Example:
        >>> result = get_market_sentiment(["bitcoin", "ethereum"])
        >>> print(result)
        {"bitcoin": {"sentiment_score": 75, "social_volume": 15000, ...}, ...}
    """
    try:
        sentiment_data = {}
        for coin_id in coin_ids:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            params = {
                "localization": False,
                "tickers": False,
                "market_data": False,
                "community_data": True,
                "developer_data": True,
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            sentiment_data[coin_id] = {
                "community_score": data.get("community_score"),
                "developer_score": data.get("developer_score"),
                "public_interest_score": data.get(
                    "public_interest_score"
                ),
                "community_data": data.get("community_data"),
                "developer_data": data.get("developer_data"),
            }

            # Rate limiting to avoid API restrictions
            time.sleep(0.6)

        return json.dumps(sentiment_data, indent=2)

    except requests.RequestException as e:
        return json.dumps(
            {"error": f"Failed to fetch market sentiment: {str(e)}"}
        )
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


# Initialize the agent with expanded tools
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    agent_description="Advanced financial advisor agent with comprehensive cryptocurrency market analysis capabilities across multiple platforms including Jupiter Protocol and HTX",
    system_prompt="You are an advanced financial advisor agent with access to real-time cryptocurrency data from multiple sources including CoinGecko, Jupiter Protocol, and HTX. You can help users analyze market trends, check prices, find trading opportunities, perform swaps, and get detailed market insights. Always provide accurate, up-to-date information and explain market data in an easy-to-understand way.",
    max_loops=1,
    max_tokens=4096,
    model_name="gpt-4o-mini",
    dynamic_temperature_enabled=True,
    output_type="all",
    tools=[
        get_coin_price,
        get_top_cryptocurrencies,
        search_cryptocurrencies,
        get_jupiter_quote,
        get_htx_market_data,
        get_token_historical_data,
        get_defi_stats,
        get_jupiter_tokens,
        get_htx_trading_pairs,
        get_market_sentiment,
    ],
    # Upload your tools to the tools parameter here!
)

# agent.run("Use defi stats to find the best defi project to invest in")
agent.run(
    "Get the price of bitcoin on both functions get_htx_crypto_price and get_crypto_price and also get the market sentiment for bitcoin"
)
# Automatically executes any number and combination of tools you have uploaded to the tools parameter!
