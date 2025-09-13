# Swarms Tools Documentation

Swarms provides a comprehensive toolkit for integrating various types of tools into your AI agents. This guide covers all available tool options including callable functions, MCP servers, schemas, and more.

## Installation

```bash
pip install swarms
```

## Overview

Swarms provides a comprehensive suite of tool integration methods to enhance your AI agents' capabilities:

| Tool Type | Description |
|-----------|-------------|
| **Callable Functions** | Direct integration of Python functions with proper type hints and comprehensive docstrings for immediate tool functionality |
| **MCP Servers** | Model Context Protocol servers enabling distributed tool functionality across multiple services and environments |
| **Tool Schemas** | Structured tool definitions that provide standardized interfaces and validation for tool integration |
| **Tool Collections** | Pre-built tool packages offering ready-to-use functionality for common use cases |

---

## Method 1: Callable Functions

Callable functions are the simplest way to add tools to your Swarms agents. They are regular Python functions with type hints and comprehensive docstrings.

### Step 1: Define Your Tool Functions

Create functions with the following requirements:

- **Type hints** for all parameters and return values

- **Comprehensive docstrings** with Args, Returns, Raises, and Examples sections

- **Error handling** for robust operation

#### Example: Cryptocurrency Price Tools

```python
import json
import requests
from swarms import Agent


def get_coin_price(coin_id: str, vs_currency: str = "usd") -> str:
    """
    Get the current price of a specific cryptocurrency.

    Args:
        coin_id (str): The CoinGecko ID of the cryptocurrency 
                      Examples: 'bitcoin', 'ethereum', 'cardano'
        vs_currency (str, optional): The target currency for price conversion.
                                   Supported: 'usd', 'eur', 'gbp', 'jpy', etc.
                                   Defaults to "usd".

    Returns:
        str: JSON formatted string containing the coin's current price and market data
             including market cap, 24h volume, and price changes

    Raises:
        requests.RequestException: If the API request fails due to network issues
        ValueError: If coin_id is empty or invalid
        TimeoutError: If the request takes longer than 10 seconds

    Example:
        >>> result = get_coin_price("bitcoin", "usd")
        >>> print(result)
        {"bitcoin": {"usd": 45000, "usd_market_cap": 850000000000, ...}}
        
        >>> result = get_coin_price("ethereum", "eur")
        >>> print(result)
        {"ethereum": {"eur": 3200, "eur_market_cap": 384000000000, ...}}
    """
    try:
        # Validate input parameters
        if not coin_id or not coin_id.strip():
            raise ValueError("coin_id cannot be empty")
            
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": coin_id.lower().strip(),
            "vs_currencies": vs_currency.lower(),
            "include_market_cap": True,
            "include_24hr_vol": True,
            "include_24hr_change": True,
            "include_last_updated_at": True,
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        
        # Check if the coin was found
        if not data:
            return json.dumps({
                "error": f"Cryptocurrency '{coin_id}' not found. Please check the coin ID."
            })
            
        return json.dumps(data, indent=2)

    except requests.RequestException as e:
        return json.dumps({
            "error": f"Failed to fetch price for {coin_id}: {str(e)}",
            "suggestion": "Check your internet connection and try again"
        })
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


def get_top_cryptocurrencies(limit: int = 10, vs_currency: str = "usd") -> str:
    """
    Fetch the top cryptocurrencies by market capitalization.

    Args:
        limit (int, optional): Number of coins to retrieve. 
                              Range: 1-250 coins
                              Defaults to 10.
        vs_currency (str, optional): The target currency for price conversion.
                                   Supported: 'usd', 'eur', 'gbp', 'jpy', etc.
                                   Defaults to "usd".

    Returns:
        str: JSON formatted string containing top cryptocurrencies with detailed market data
             including: id, symbol, name, current_price, market_cap, market_cap_rank,
             total_volume, price_change_24h, price_change_7d, last_updated

    Raises:
        requests.RequestException: If the API request fails
        ValueError: If limit is not between 1 and 250

    Example:
        >>> result = get_top_cryptocurrencies(5, "usd")
        >>> print(result)
        [{"id": "bitcoin", "name": "Bitcoin", "current_price": 45000, ...}]
        
        >>> result = get_top_cryptocurrencies(limit=3, vs_currency="eur")
        >>> print(result)
        [{"id": "bitcoin", "name": "Bitcoin", "current_price": 38000, ...}]
    """
    try:
        # Validate parameters
        if not isinstance(limit, int) or not 1 <= limit <= 250:
            raise ValueError("Limit must be an integer between 1 and 250")

        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": vs_currency.lower(),
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1,
            "sparkline": False,
            "price_change_percentage": "24h,7d",
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Simplify and structure the data for better readability
        simplified_data = []
        for coin in data:
            simplified_data.append({
                "id": coin.get("id"),
                "symbol": coin.get("symbol", "").upper(),
                "name": coin.get("name"),
                "current_price": coin.get("current_price"),
                "market_cap": coin.get("market_cap"),
                "market_cap_rank": coin.get("market_cap_rank"),
                "total_volume": coin.get("total_volume"),
                "price_change_24h": round(coin.get("price_change_percentage_24h", 0), 2),
                "price_change_7d": round(coin.get("price_change_percentage_7d_in_currency", 0), 2),
                "last_updated": coin.get("last_updated"),
            })

        return json.dumps(simplified_data, indent=2)

    except (requests.RequestException, ValueError) as e:
        return json.dumps({
            "error": f"Failed to fetch top cryptocurrencies: {str(e)}"
        })
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


def search_cryptocurrencies(query: str) -> str:
    """
    Search for cryptocurrencies by name or symbol.

    Args:
        query (str): The search term (coin name or symbol)
                    Examples: 'bitcoin', 'btc', 'ethereum', 'eth'
                    Case-insensitive search

    Returns:
        str: JSON formatted string containing search results with coin details
             including: id, name, symbol, market_cap_rank, thumb (icon URL)
             Limited to top 10 results for performance

    Raises:
        requests.RequestException: If the API request fails
        ValueError: If query is empty

    Example:
        >>> result = search_cryptocurrencies("ethereum")
        >>> print(result)
        {"coins": [{"id": "ethereum", "name": "Ethereum", "symbol": "eth", ...}]}
        
        >>> result = search_cryptocurrencies("btc")
        >>> print(result)
        {"coins": [{"id": "bitcoin", "name": "Bitcoin", "symbol": "btc", ...}]}
    """
    try:
        # Validate input
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")
            
        url = "https://api.coingecko.com/api/v3/search"
        params = {"query": query.strip()}

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Extract and format the results
        coins = data.get("coins", [])[:10]  # Limit to top 10 results
        
        result = {
            "coins": coins,
            "query": query,
            "total_results": len(data.get("coins", [])),
            "showing": min(len(coins), 10)
        }

        return json.dumps(result, indent=2)

    except requests.RequestException as e:
        return json.dumps({
            "error": f'Failed to search for "{query}": {str(e)}'
        })
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})
```

### Step 2: Configure Your Agent

Create an agent with the following key parameters:

```python
# Initialize the agent with cryptocurrency tools
agent = Agent(
    agent_name="Financial-Analysis-Agent",                    # Unique identifier for your agent
    agent_description="Personal finance advisor agent with cryptocurrency market analysis capabilities",
    system_prompt="""You are a personal finance advisor agent with access to real-time 
    cryptocurrency data from CoinGecko. You can help users analyze market trends, check 
    coin prices, find trending cryptocurrencies, and search for specific coins. Always 
    provide accurate, up-to-date information and explain market data in an easy-to-understand way.""",
    max_loops=1,                                              # Number of reasoning loops
    max_tokens=4096,                                          # Maximum response length
    model_name="anthropic/claude-3-opus-20240229",          # LLM model to use
    dynamic_temperature_enabled=True,                         # Enable adaptive creativity
    output_type="all",                                        # Return complete response
    tools=[                                                   # List of callable functions
        get_coin_price,
        get_top_cryptocurrencies,
        search_cryptocurrencies,
    ],
)
```

### Step 3: Use Your Agent

```python
# Example usage with different queries
response = agent.run("What are the top 5 cryptocurrencies by market cap?")
print(response)

# Query with specific parameters
response = agent.run("Get the current price of Bitcoin and Ethereum in EUR")
print(response)

# Search functionality
response = agent.run("Search for cryptocurrencies related to 'cardano'")
print(response)
```

---

## Method 2: MCP (Model Context Protocol) Servers

MCP servers provide a standardized way to create distributed tool functionality. They're ideal for:

- **Reusable tools** across multiple agents

- **Complex tool logic** that needs isolation

- **Third-party tool integration**

- **Scalable architectures**

### Step 1: Create Your MCP Server

```python
from mcp.server.fastmcp import FastMCP
import requests

# Initialize the MCP server with configuration
mcp = FastMCP("OKXCryptoPrice")  # Server name for identification
mcp.settings.port = 8001         # Port for server communication
```

### Step 2: Define MCP Tools

Each MCP tool requires the `@mcp.tool` decorator with specific parameters:

```python
@mcp.tool(
    name="get_okx_crypto_price",                              # Tool identifier (must be unique)
    description="Get the current price and basic information for a given cryptocurrency from OKX exchange.",
)
def get_okx_crypto_price(symbol: str) -> str:
    """
    Get the current price and basic information for a given cryptocurrency using OKX API.

    Args:
        symbol (str): The cryptocurrency trading pair
                     Format: 'BASE-QUOTE' (e.g., 'BTC-USDT', 'ETH-USDT')
                     If only base currency provided, '-USDT' will be appended
                     Case-insensitive input

    Returns:
        str: A formatted string containing:
             - Current price in USDT
             - 24-hour price change percentage
             - Formatted for human readability

    Raises:
        requests.RequestException: If the OKX API request fails
        ValueError: If symbol format is invalid
        ConnectionError: If unable to connect to OKX servers

    Example:
        >>> get_okx_crypto_price('BTC-USDT')
        'Current price of BTC/USDT: $45,000.00\n24h Change: +2.34%'
        
        >>> get_okx_crypto_price('eth')  # Automatically converts to ETH-USDT
        'Current price of ETH/USDT: $3,200.50\n24h Change: -1.23%'
    """
    try:
        # Input validation and formatting
        if not symbol or not symbol.strip():
            return "Error: Please provide a valid trading pair (e.g., 'BTC-USDT')"

        # Normalize symbol format
        symbol = symbol.upper().strip()
        if not symbol.endswith("-USDT"):
            symbol = f"{symbol}-USDT"

        # OKX API endpoint for ticker information
        url = f"https://www.okx.com/api/v5/market/ticker?instId={symbol}"

        # Make the API request with timeout
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Check API response status
        if data.get("code") != "0":
            return f"Error: {data.get('msg', 'Unknown error from OKX API')}"

        # Extract ticker data
        ticker_data = data.get("data", [{}])[0]
        if not ticker_data:
            return f"Error: Could not find data for {symbol}. Please verify the trading pair exists."

        # Parse numerical data
        price = float(ticker_data.get("last", 0))
        change_percent = float(ticker_data.get("change24h", 0)) * 100  # Convert to percentage

        # Format response
        base_currency = symbol.split("-")[0]
        change_symbol = "+" if change_percent >= 0 else ""
        
        return (f"Current price of {base_currency}/USDT: ${price:,.2f}\n"
                f"24h Change: {change_symbol}{change_percent:.2f}%")

    except requests.exceptions.Timeout:
        return "Error: Request timed out. OKX servers may be slow."
    except requests.exceptions.RequestException as e:
        return f"Error fetching OKX data: {str(e)}"
    except (ValueError, KeyError) as e:
        return f"Error parsing OKX response: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


@mcp.tool(
    name="get_okx_crypto_volume",                             # Second tool with different functionality
    description="Get the 24-hour trading volume for a given cryptocurrency from OKX exchange.",
)
def get_okx_crypto_volume(symbol: str) -> str:
    """
    Get the 24-hour trading volume for a given cryptocurrency using OKX API.

    Args:
        symbol (str): The cryptocurrency trading pair
                     Format: 'BASE-QUOTE' (e.g., 'BTC-USDT', 'ETH-USDT')
                     If only base currency provided, '-USDT' will be appended
                     Case-insensitive input

    Returns:
        str: A formatted string containing:
             - 24-hour trading volume in the base currency
             - Volume formatted with thousand separators
             - Currency symbol for clarity

    Raises:
        requests.RequestException: If the OKX API request fails
        ValueError: If symbol format is invalid

    Example:
        >>> get_okx_crypto_volume('BTC-USDT')
        '24h Trading Volume for BTC/USDT: 12,345.67 BTC'
        
        >>> get_okx_crypto_volume('ethereum')  # Converts to ETH-USDT
        '24h Trading Volume for ETH/USDT: 98,765.43 ETH'
    """
    try:
        # Input validation and formatting
        if not symbol or not symbol.strip():
            return "Error: Please provide a valid trading pair (e.g., 'BTC-USDT')"

        # Normalize symbol format
        symbol = symbol.upper().strip()
        if not symbol.endswith("-USDT"):
            symbol = f"{symbol}-USDT"

        # OKX API endpoint
        url = f"https://www.okx.com/api/v5/market/ticker?instId={symbol}"

        # Make API request
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Validate API response
        if data.get("code") != "0":
            return f"Error: {data.get('msg', 'Unknown error from OKX API')}"

        ticker_data = data.get("data", [{}])[0]
        if not ticker_data:
            return f"Error: Could not find data for {symbol}. Please verify the trading pair."

        # Extract volume data
        volume_24h = float(ticker_data.get("vol24h", 0))
        base_currency = symbol.split("-")[0]
        
        return f"24h Trading Volume for {base_currency}/USDT: {volume_24h:,.2f} {base_currency}"

    except requests.exceptions.RequestException as e:
        return f"Error fetching OKX data: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"
```

### Step 3: Start Your MCP Server

```python
if __name__ == "__main__":
    # Run the MCP server with SSE (Server-Sent Events) transport
    # Server will be available at http://localhost:8001/mcp
    mcp.run(transport="sse")
```

### Step 4: Connect Agent to MCP Server

```python
from swarms import Agent

# Method 2: Using direct URL (simpler for development)
mcp_url = "http://0.0.0.0:8001/mcp"

# Initialize agent with MCP tools
agent = Agent(
    agent_name="Financial-Analysis-Agent",                    # Agent identifier
    agent_description="Personal finance advisor with OKX exchange data access",
    system_prompt="""You are a financial analysis agent with access to real-time 
    cryptocurrency data from OKX exchange. You can check prices, analyze trading volumes, 
    and provide market insights. Always format numerical data clearly and explain 
    market movements in context.""",
    max_loops=1,                                              # Processing loops
    mcp_url=mcp_url,                                         # MCP server connection
    output_type="all",                                        # Complete response format
    # Note: tools are automatically loaded from MCP server
)
```

### Step 5: Use Your MCP-Enabled Agent

```python
# The agent automatically discovers and uses tools from the MCP server
response = agent.run(
    "Fetch the price for Bitcoin using the OKX exchange and also get its trading volume"
)
print(response)

# Multiple tool usage
response = agent.run(
    "Compare the prices of BTC, ETH, and ADA on OKX, and show their trading volumes"
)
print(response)
```

---

## Best Practices

### Function Design

| Practice | Description |
|----------|-------------|
| Type Hints | Always use type hints for all parameters and return values |
| Docstrings | Write comprehensive docstrings with Args, Returns, Raises, and Examples |
| Error Handling | Implement proper error handling with specific exception types |
| Input Validation | Validate input parameters before processing |
| Data Structure | Return structured data (preferably JSON) for consistency |

### MCP Server Development

| Practice | Description |
|----------|-------------|
| Tool Naming | Use descriptive tool names that clearly indicate functionality |
| Timeouts | Set appropriate timeouts for external API calls |
| Error Handling | Implement graceful error handling for network issues |
| Configuration | Use environment variables for sensitive configuration |
| Testing | Test tools independently before integration |

### Agent Configuration

| Practice | Description |
|----------|-------------|
| Loop Control | Choose appropriate max_loops based on task complexity |
| Token Management | Set reasonable token limits to control response length |
| System Prompts | Write clear system prompts that explain tool capabilities |
| Agent Naming | Use meaningful agent names for debugging and logging |
| Tool Integration | Consider tool combinations for comprehensive functionality |

### Performance Optimization

| Practice | Description |
|----------|-------------|
| Data Caching | Cache frequently requested data when possible |
| Connection Management | Use connection pooling for multiple API calls |
| Rate Control | Implement rate limiting to respect API constraints |
| Performance Monitoring | Monitor tool execution times and optimize slow operations |
| Async Operations | Use async operations for concurrent tool execution when supported |

---

## Troubleshooting

### Common Issues

#### Tool Not Found

```python
# Ensure function is in tools list
agent = Agent(
    # ... other config ...
    tools=[your_function_name],  # Function object, not string
)
```

#### MCP Connection Failed
```python
# Check server status and URL
import requests
response = requests.get("http://localhost:8001/health")  # Health check endpoint
```

#### Type Hint Errors

```python
# Always specify return types
def my_tool(param: str) -> str:  # Not just -> None
    return "result"
```

#### JSON Parsing Issues

```python
# Always return valid JSON strings
import json
return json.dumps({"result": data}, indent=2)
```