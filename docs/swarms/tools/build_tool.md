# Swarms Tool Documentation

A tool is a Python function designed to perform specific tasks with clear type annotations and comprehensive docstrings. Below are examples of financial tools to help you get started.

## Rules

To create a tool in the Swarms environment, follow these rules:

1. **Function Definition**: 
   - The tool must be defined as a Python function.
   - The function should perform a specific task and be named appropriately.

2. **Type Annotations**: 
   - All arguments and the return value must have type annotations.
   - Both input and output types must be strings (`str`).

3. **Docstrings**: 
   - Each function must include a comprehensive docstring that adheres to PEP 257 standards. The docstring should explain:
     - The purpose of the function.
     - Arguments: names, types, and descriptions.
     - Return value: type and description.
     - Potential exceptions that the function may raise.

4. **Input and Output Types**:
   - The function's input must be a string.
   - The function's output must be a string.

## Example Financial Tools

### Example 1: Fetch Stock Price from Yahoo Finance

```python
import yfinance as yf

def get_stock_price(symbol: str) -> str:
    """
    Fetches the current stock price from Yahoo Finance.

    Args:
        symbol (str): The stock symbol (e.g., "AAPL", "TSLA", "NVDA").

    Returns:
        str: A formatted string containing the current stock price and basic information.

    Raises:
        ValueError: If the stock symbol is invalid or data cannot be retrieved.
        Exception: If there is an error with the API request.
    """
    try:
        # Remove any whitespace and convert to uppercase
        symbol = symbol.strip().upper()
        
        if not symbol:
            raise ValueError("Stock symbol cannot be empty.")
        
        # Fetch stock data
        stock = yf.Ticker(symbol)
        info = stock.info
        
        if not info or 'regularMarketPrice' not in info:
            raise ValueError(f"Unable to fetch data for symbol: {symbol}")
        
        current_price = info.get('regularMarketPrice', 'N/A')
        previous_close = info.get('regularMarketPreviousClose', 'N/A')
        market_cap = info.get('marketCap', 'N/A')
        company_name = info.get('longName', symbol)
        
        # Format market cap for readability
        if isinstance(market_cap, (int, float)) and market_cap > 0:
            if market_cap >= 1e12:
                market_cap_str = f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                market_cap_str = f"${market_cap/1e9:.2f}B"
            elif market_cap >= 1e6:
                market_cap_str = f"${market_cap/1e6:.2f}M"
            else:
                market_cap_str = f"${market_cap:,.0f}"
        else:
            market_cap_str = "N/A"
        
        # Calculate price change
        if isinstance(current_price, (int, float)) and isinstance(previous_close, (int, float)):
            price_change = current_price - previous_close
            price_change_percent = (price_change / previous_close) * 100
            change_str = f"{price_change:+.2f} ({price_change_percent:+.2f}%)"
        else:
            change_str = "N/A"
        
        result = f"""
Stock: {company_name} ({symbol})
Current Price: ${current_price}
Previous Close: ${previous_close}
Change: {change_str}
Market Cap: {market_cap_str}
        """.strip()
        
        return result
        
    except ValueError as e:
        print(f"Value Error: {e}")
        raise
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        raise
```

### Example 2: Fetch Cryptocurrency Price from CoinGecko

```python
import requests

def get_crypto_price(coin_id: str) -> str:
    """
    Fetches the current cryptocurrency price from CoinGecko API.

    Args:
        coin_id (str): The cryptocurrency ID (e.g., "bitcoin", "ethereum", "cardano").

    Returns:
        str: A formatted string containing the current crypto price and market data.

    Raises:
        ValueError: If the coin ID is invalid or data cannot be retrieved.
        requests.exceptions.RequestException: If there is an error with the API request.
    """
    try:
        # Remove any whitespace and convert to lowercase
        coin_id = coin_id.strip().lower()
        
        if not coin_id:
            raise ValueError("Coin ID cannot be empty.")
        
        url = f"https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": coin_id,
            "vs_currencies": "usd",
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
            "include_last_updated_at": "true"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if coin_id not in data:
            raise ValueError(f"Coin ID '{coin_id}' not found. Please check the spelling.")
        
        coin_data = data[coin_id]
        
        if not coin_data:
            raise ValueError(f"No data available for coin ID: {coin_id}")
        
        usd_price = coin_data.get('usd', 'N/A')
        market_cap = coin_data.get('usd_market_cap', 'N/A')
        volume_24h = coin_data.get('usd_24h_vol', 'N/A')
        change_24h = coin_data.get('usd_24h_change', 'N/A')
        last_updated = coin_data.get('last_updated_at', 'N/A')
        
        # Format large numbers for readability
        def format_number(value):
            if isinstance(value, (int, float)) and value > 0:
                if value >= 1e12:
                    return f"${value/1e12:.2f}T"
                elif value >= 1e9:
                    return f"${value/1e9:.2f}B"
                elif value >= 1e6:
                    return f"${value/1e6:.2f}M"
                elif value >= 1e3:
                    return f"${value/1e3:.2f}K"
                else:
                    return f"${value:,.2f}"
            return "N/A"
        
        # Format the result
        result = f"""
Cryptocurrency: {coin_id.title()}
Current Price: ${usd_price:,.8f}" if isinstance(usd_price, (int, float)) else f"Current Price: {usd_price}
Market Cap: {format_number(market_cap)}
24h Volume: {format_number(volume_24h)}
24h Change: {change_24h:+.2f}%" if isinstance(change_24h, (int, float)) else f"24h Change: {change_24h}
Last Updated: {last_updated}
        """.strip()
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        raise
    except ValueError as e:
        print(f"Value Error: {e}")
        raise
    except Exception as e:
        print(f"Error fetching crypto data: {e}")
        raise
```

### Example 3: Calculate Portfolio Performance

```python
def calculate_portfolio_performance(initial_investment_str: str, current_value_str: str, time_period_str: str) -> str:
    """
    Calculates portfolio performance metrics including return percentage and annualized return.

    Args:
        initial_investment_str (str): The initial investment amount as a string.
        current_value_str (str): The current portfolio value as a string.
        time_period_str (str): The time period in years as a string.

    Returns:
        str: A formatted string containing portfolio performance metrics.

    Raises:
        ValueError: If any of the inputs cannot be converted to the appropriate type or are negative.
    """
    try:
        initial_investment = float(initial_investment_str)
        current_value = float(current_value_str)
        time_period = float(time_period_str)
        
        if initial_investment <= 0 or current_value < 0 or time_period <= 0:
            raise ValueError("Initial investment and time period must be positive, current value must be non-negative.")
        
        # Calculate total return
        total_return = current_value - initial_investment
        total_return_percentage = (total_return / initial_investment) * 100
        
        # Calculate annualized return
        if time_period > 0:
            annualized_return = ((current_value / initial_investment) ** (1 / time_period) - 1) * 100
        else:
            annualized_return = 0
        
        # Determine performance status
        if total_return > 0:
            status = "Profitable"
        elif total_return < 0:
            status = "Loss"
        else:
            status = "Break-even"
        
        result = f"""
Portfolio Performance Analysis:
Initial Investment: ${initial_investment:,.2f}
Current Value: ${current_value:,.2f}
Time Period: {time_period:.1f} years

Total Return: ${total_return:+,.2f} ({total_return_percentage:+.2f}%)
Annualized Return: {annualized_return:+.2f}%
Status: {status}
        """.strip()
        
        return result
        
    except ValueError as e:
        print(f"Value Error: {e}")
        raise
    except Exception as e:
        print(f"Error calculating portfolio performance: {e}")
        raise
```

### Example 4: Calculate Compound Interest

```python
def calculate_compound_interest(principal_str: str, rate_str: str, time_str: str, compounding_frequency_str: str) -> str:
    """
    Calculates compound interest for investment planning.

    Args:
        principal_str (str): The initial investment amount as a string.
        rate_str (str): The annual interest rate (as decimal) as a string.
        time_str (str): The investment time period in years as a string.
        compounding_frequency_str (str): The number of times interest is compounded per year as a string.

    Returns:
        str: A formatted string containing the compound interest calculation results.

    Raises:
        ValueError: If any of the inputs cannot be converted to the appropriate type or are negative.
    """
    try:
        principal = float(principal_str)
        rate = float(rate_str)
        time = float(time_str)
        n = int(compounding_frequency_str)
        
        if principal <= 0 or rate < 0 or time <= 0 or n <= 0:
            raise ValueError("Principal, time, and compounding frequency must be positive. Rate must be non-negative.")
        
        # Calculate compound interest
        amount = principal * (1 + rate / n) ** (n * time)
        interest_earned = amount - principal
        
        # Calculate effective annual rate
        effective_rate = ((1 + rate / n) ** n - 1) * 100
        
        result = f"""
Compound Interest Calculation:
Principal: ${principal:,.2f}
Annual Rate: {rate*100:.2f}%
Time Period: {time:.1f} years
Compounding Frequency: {n} times per year

Final Amount: ${amount:,.2f}
Interest Earned: ${interest_earned:,.2f}
Effective Annual Rate: {effective_rate:.2f}%
        """.strip()
        
        return result
        
    except ValueError as e:
        print(f"Value Error: {e}")
        raise
    except Exception as e:
        print(f"Error calculating compound interest: {e}")
        raise
```

## Integrating Tools into an Agent

To integrate tools into an agent, simply pass callable functions with proper type annotations and documentation into the agent class.

```python
from swarms import Agent

# Initialize the financial analysis agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    system_prompt=(
        "You are a professional financial analyst agent. Use the provided tools to "
        "analyze stocks, cryptocurrencies, and investment performance. Provide "
        "clear, accurate financial insights and recommendations. Always format "
        "responses in markdown for better readability."
    ),
    model_name="gpt-4.1",
    max_loops=3,
    autosave=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    dynamic_temperature_enabled=True,
    saved_state_path="financial_agent.json",
    tools=[get_stock_price, get_crypto_price, calculate_portfolio_performance],
    user_name="financial_analyst",
    retry_attempts=3,
    context_length=200000,
)

# Run the agent
response = agent("Analyze the current price of Apple stock and Bitcoin, then calculate the performance of a $10,000 investment in each over the past 2 years.")
print(response)
```

## Complete Financial Analysis Example

```python
import yfinance as yf
import requests
from swarms import Agent

def get_stock_price(symbol: str) -> str:
    """
    Fetches the current stock price from Yahoo Finance.

    Args:
        symbol (str): The stock symbol (e.g., "AAPL", "TSLA", "NVDA").

    Returns:
        str: A formatted string containing the current stock price and basic information.

    Raises:
        ValueError: If the stock symbol is invalid or data cannot be retrieved.
        Exception: If there is an error with the API request.
    """
    try:
        symbol = symbol.strip().upper()
        
        if not symbol:
            raise ValueError("Stock symbol cannot be empty.")
        
        stock = yf.Ticker(symbol)
        info = stock.info
        
        if not info or 'regularMarketPrice' not in info:
            raise ValueError(f"Unable to fetch data for symbol: {symbol}")
        
        current_price = info.get('regularMarketPrice', 'N/A')
        previous_close = info.get('regularMarketPreviousClose', 'N/A')
        market_cap = info.get('marketCap', 'N/A')
        company_name = info.get('longName', symbol)
        
        if isinstance(market_cap, (int, float)) and market_cap > 0:
            if market_cap >= 1e12:
                market_cap_str = f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                market_cap_str = f"${market_cap/1e9:.2f}B"
            elif market_cap >= 1e6:
                market_cap_str = f"${market_cap/1e6:.2f}M"
            else:
                market_cap_str = f"${market_cap:,.0f}"
        else:
            market_cap_str = "N/A"
        
        if isinstance(current_price, (int, float)) and isinstance(previous_close, (int, float)):
            price_change = current_price - previous_close
            price_change_percent = (price_change / previous_close) * 100
            change_str = f"{price_change:+.2f} ({price_change_percent:+.2f}%)"
        else:
            change_str = "N/A"
        
        result = f"""
Stock: {company_name} ({symbol})
Current Price: ${current_price}
Previous Close: ${previous_close}
Change: {change_str}
Market Cap: {market_cap_str}
        """.strip()
        
        return result
        
    except ValueError as e:
        print(f"Value Error: {e}")
        raise
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        raise

def get_crypto_price(coin_id: str) -> str:
    """
    Fetches the current cryptocurrency price from CoinGecko API.

    Args:
        coin_id (str): The cryptocurrency ID (e.g., "bitcoin", "ethereum", "cardano").

    Returns:
        str: A formatted string containing the current crypto price and market data.

    Raises:
        ValueError: If the coin ID is invalid or data cannot be retrieved.
        requests.exceptions.RequestException: If there is an error with the API request.
    """
    try:
        coin_id = coin_id.strip().lower()
        
        if not coin_id:
            raise ValueError("Coin ID cannot be empty.")
        
        url = f"https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": coin_id,
            "vs_currencies": "usd",
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
            "include_last_updated_at": "true"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if coin_id not in data:
            raise ValueError(f"Coin ID '{coin_id}' not found. Please check the spelling.")
        
        coin_data = data[coin_id]
        
        if not coin_data:
            raise ValueError(f"No data available for coin ID: {coin_id}")
        
        usd_price = coin_data.get('usd', 'N/A')
        market_cap = coin_data.get('usd_market_cap', 'N/A')
        volume_24h = coin_data.get('usd_24h_vol', 'N/A')
        change_24h = coin_data.get('usd_24h_change', 'N/A')
        last_updated = coin_data.get('last_updated_at', 'N/A')
        
        def format_number(value):
            if isinstance(value, (int, float)) and value > 0:
                if value >= 1e12:
                    return f"${value/1e12:.2f}T"
                elif value >= 1e9:
                    return f"${value/1e9:.2f}B"
                elif value >= 1e6:
                    return f"${value/1e6:.2f}M"
                elif value >= 1e3:
                    return f"${value/1e3:.2f}K"
                else:
                    return f"${value:,.2f}"
            return "N/A"
        
        result = f"""
Cryptocurrency: {coin_id.title()}
Current Price: ${usd_price:,.8f}" if isinstance(usd_price, (int, float)) else f"Current Price: {usd_price}
Market Cap: {format_number(market_cap)}
24h Volume: {format_number(volume_24h)}
24h Change: {change_24h:+.2f}%" if isinstance(change_24h, (int, float)) else f"24h Change: {change_24h}
Last Updated: {last_updated}
        """.strip()
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        raise
    except ValueError as e:
        print(f"Value Error: {e}")
        raise
    except Exception as e:
        print(f"Error fetching crypto data: {e}")
        raise

# Initialize the financial analysis agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    system_prompt=(
        "You are a professional financial analyst agent specializing in stock and "
        "cryptocurrency analysis. Use the provided tools to fetch real-time market "
        "data and provide comprehensive financial insights. Always present data "
        "in a clear, professional format with actionable recommendations."
    ),
    model_name="gpt-4.1",
    max_loops=3,
    autosave=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    dynamic_temperature_enabled=True,
    saved_state_path="financial_agent.json",
    tools=[get_stock_price, get_crypto_price],
    user_name="financial_analyst",
    retry_attempts=3,
    context_length=200000,
)

# Run the agent
response = agent("What are the current prices and market data for Apple stock and Bitcoin? Provide a brief analysis of their performance.")
print(response)
```
