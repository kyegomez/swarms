import json
import requests
from swarms.tools.py_func_to_openai_func_str import (
    convert_multiple_functions_to_openai_function_schema,
)


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


funcs = [
    get_coin_price,
    get_top_cryptocurrencies,
    search_cryptocurrencies,
]

print(
    json.dumps(
        convert_multiple_functions_to_openai_function_schema(funcs),
        indent=2,
    )
)
