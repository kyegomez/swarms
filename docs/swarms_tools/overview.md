# Swarms Tools


Welcome to **Swarms Tools**, the ultimate package for integrating **cutting-edge APIs** into Python functions with seamless multi-agent system compatibility. Designed for enterprises at the forefront of innovation, **Swarms Tools** is your key to simplifying complexity and unlocking operational excellence.

---

## 🚀 Features

- **Unified API Integration**: Ready-to-use Python functions for financial data, social media, IoT, and more.
- **Enterprise-Grade Design**: Comprehensive type hints, structured outputs, and robust documentation.
- **Agent-Ready Framework**: Optimized for seamless integration into Swarms' multi-agent orchestration systems.
- **Expandable Architecture**: Easily extend functionality with a standardized schema for new tools.

---

## 🔧 Installation

```bash
pip3 install -U swarms-tools
```

---

## 📂 Directory Structure

```plaintext
swarms-tools/
├── swarms_tools/
│   ├── finance/
│   │   ├── htx_tool.py
│   │   ├── eodh_api.py
│   │   └── coingecko_tool.py
│   ├── social_media/
│   │   ├── telegram_tool.py
│   ├── utilities/
│   │   └── logging.py
├── tests/
│   ├── test_financial_data.py
│   └── test_social_media.py
└── README.md
```

---

## 💼 Use Cases



## Finance

Explore our diverse range of financial tools, designed to streamline your operations. If you need a tool not listed, feel free to submit an issue or accelerate integration by contributing a pull request with your tool of choice.

| **Tool Name**             | **Function**             | **Description**                                                                 |
|---------------------------|--------------------------|---------------------------------------------------------------------------------|
| `fetch_stock_news`        | `fetch_stock_news`       | Fetches the latest stock news and updates.                                     |
| `fetch_htx_data`          | `fetch_htx_data`         | Retrieves financial data from the HTX platform.                                |
| `yahoo_finance_api`       | `yahoo_finance_api`      | Fetches comprehensive stock data from Yahoo Finance, including prices and trends. |
| `coin_gecko_coin_api`     | `coin_gecko_coin_api`    | Fetches cryptocurrency data from CoinGecko, including market and price information. |
| `helius_api_tool`         | `helius_api_tool`        | Retrieves blockchain account, transaction, or token data using the Helius API. |
| `okx_api_tool`            | `okx_api_tool`           | Fetches detailed cryptocurrency data for coins from the OKX exchange.         |


### Financial Data Retrieval
Enable precise and actionable financial insights:

#### Example 1: Fetch Historical Data
```python
from swarms_tools import fetch_htx_data

# Fetch historical trading data for "Swarms Corporation"
response = fetch_htx_data("swarms")
print(response)
```

#### Example 2: Stock News Analysis
```python
from swarms_tools import fetch_stock_news

# Retrieve latest stock news for Apple
news = fetch_stock_news("AAPL")
print(news)
```

#### Example 3: Cryptocurrency Metrics
```python
from swarms_tools import coin_gecko_coin_api

# Fetch live data for Bitcoin
crypto_data = coin_gecko_coin_api("bitcoin")
print(crypto_data)
```

### Social Media Automation
Streamline communication and engagement:

#### Example: Telegram Bot Messaging
```python
from swarms_tools import telegram_dm_or_tag_api

def send_alert(response: str):
    telegram_dm_or_tag_api(response)

# Send a message to a user or group
send_alert("Mission-critical update from Swarms.")
```

---

## Dex Screener

This is a tool that allows you to fetch data from the Dex Screener API. It supports multiple chains and multiple tokens.

```python
from swarms_tools.finance.dex_screener import (
    fetch_latest_token_boosts,
    fetch_dex_screener_profiles,
)


fetch_dex_screener_profiles()
fetch_latest_token_boosts()

```

---


## Structs
The tool chainer enables the execution of multiple tools in a sequence, allowing for the aggregation of their results in either a parallel or sequential manner.

```python
# Example usage
from loguru import logger

from swarms_tools.structs import tool_chainer


if __name__ == "__main__":
    logger.add("tool_chainer.log", rotation="500 MB", level="INFO")

    # Example tools
    def tool1():
        return "Tool1 Result"

    def tool2():
        return "Tool2 Result"

    # def tool3():
    #     raise ValueError("Simulated error in Tool3")

    tools = [tool1, tool2]

    # Parallel execution
    parallel_results = tool_chainer(tools, parallel=True)
    print("Parallel Results:", parallel_results)

    # Sequential execution
    # sequential_results = tool_chainer(tools, parallel=False)
    # print("Sequential Results:", sequential_results)

```
---

## 🧩 Standardized Schema

Every tool in **Swarms Tools** adheres to a strict schema for maintainability and interoperability:

### Schema Template

1. **Functionality**:
   - Encapsulate API logic into a modular, reusable function.

2. **Typing**:
   - Leverage Python type hints for input validation and clarity.

   Example:
   ```python
   def fetch_data(symbol: str, date_range: str) -> str:
       """
       Fetch financial data for a given symbol and date range.

       Args:
           symbol (str): Ticker symbol of the asset.
           date_range (str): Timeframe for the data (e.g., '1d', '1m', '1y').

       Returns:
           dict: A dictionary containing financial metrics.
       """
       pass
   ```

3. **Documentation**:
   - Include detailed docstrings with parameter explanations and usage examples.

4. **Output Standardization**:
   - Ensure consistent outputs (e.g., strings) for easy downstream agent integration.

5. **API-Key Management**:
    - All API keys must be fetched with `os.getenv("YOUR_KEY")`


---

## 📖 Documentation

Comprehensive documentation is available to guide developers and enterprises. Visit our [official docs](https://docs.swarms.world) for detailed API references, usage examples, and best practices.

---

## 🛠 Contributing

We welcome contributions from the global developer community. To contribute:

1. **Fork the Repository**: Start by forking the repository.
2. **Create a Feature Branch**: Use a descriptive branch name: `feature/add-new-tool`.
3. **Commit Your Changes**: Write meaningful commit messages.
4. **Submit a Pull Request**: Open a pull request for review.

---

## 🛡️ License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🌠 Join the Future

Explore the limitless possibilities of agent-based systems. Together, we can build a smarter, faster, and more interconnected world.

**Visit us:** [Swarms Corporation](https://swarms.ai)  
**Follow us:** [Twitter](https://twitter.com/swarms_corp)

---

**"The future belongs to those who dare to automate it."**  
**— The Swarms Corporation**

