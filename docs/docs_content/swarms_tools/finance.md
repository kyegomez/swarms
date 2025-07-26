# Swarms Finance Tools Documentation

## Installation

```bash
pip3 install -U swarms-tools yfinance requests httpx pandas loguru backoff web3 solana spl-token
```

## Environment Variables

Create a `.env` file in your project root with the following variables (as needed):

| Environment Variable | Description | Required For |
|---------------------|-------------|--------------|
| `COINBASE_API_KEY` | Coinbase API Key | Coinbase Trading |
| `COINBASE_API_SECRET` | Coinbase API Secret | Coinbase Trading |
| `COINBASE_API_PASSPHRASE` | Coinbase API Passphrase | Coinbase Trading |
| `COINMARKETCAP_API_KEY` | CoinMarketCap API Key | CoinMarketCap Data |
| `HELIUS_API_KEY` | Helius API Key | Solana Data |
| `EODHD_API_KEY` | EODHD API Key | Stock News |
| `OKX_API_KEY` | OKX API Key | OKX Trading |
| `OKX_API_SECRET` | OKX API Secret | OKX Trading |
| `OKX_PASSPHRASE` | OKX Passphrase | OKX Trading |

## Tools Overview

| Tool | Description | Requires API Key |
|------|-------------|-----------------|
| Yahoo Finance | Real-time stock market data | No |
| CoinGecko | Cryptocurrency market data | No |
| Coinbase | Cryptocurrency trading and data | Yes |
| CoinMarketCap | Cryptocurrency market data | Yes |
| Helius | Solana blockchain data | Yes |
| DexScreener | DEX trading pairs and data | No |
| HTX (Huobi) | Cryptocurrency exchange data | No |
| OKX | Cryptocurrency exchange data | Yes |
| EODHD | Stock market news | Yes |
| Jupiter | Solana DEX aggregator | No |
| Sector Analysis | GICS sector ETF analysis | No |
| Solana Tools | Solana wallet and token tools | Yes |

## Detailed Documentation

### Yahoo Finance API

Fetch real-time and historical stock market data.

```python
from swarms_tools.finance import yahoo_finance_api

# Fetch data for single stock
data = yahoo_finance_api(["AAPL"])

# Fetch data for multiple stocks
data = yahoo_finance_api(["AAPL", "GOOG", "MSFT"])
```

**Arguments:**

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| stock_symbols | List[str] | List of stock symbols | Yes |

### CoinGecko API

Fetch comprehensive cryptocurrency data.

```python
from swarms_tools.finance import coin_gecko_coin_api

# Fetch Bitcoin data
data = coin_gecko_coin_api("bitcoin")
```

**Arguments:**

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| coin | str | Cryptocurrency ID (e.g., 'bitcoin') | Yes |

### Coinbase Trading

Execute trades and fetch market data from Coinbase.

```python
from swarms_tools.finance import get_coin_data, place_buy_order, place_sell_order

# Fetch coin data
data = get_coin_data("BTC-USD")

# Place orders
buy_order = place_buy_order("BTC-USD", amount=100)  # Buy $100 worth of BTC
sell_order = place_sell_order("BTC-USD", amount=0.01)  # Sell 0.01 BTC
```

**Arguments:**

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| symbol | str | Trading pair (e.g., 'BTC-USD') | Yes |
| amount | Union[str, float, Decimal] | Trade amount | Yes |
| sandbox | bool | Use sandbox environment | No |

### CoinMarketCap API

Fetch cryptocurrency market data from CoinMarketCap.

```python
from swarms_tools.finance import coinmarketcap_api

# Fetch single coin data
data = coinmarketcap_api(["Bitcoin"])

# Fetch multiple coins
data = coinmarketcap_api(["Bitcoin", "Ethereum", "Tether"])
```

**Arguments:**

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| coin_names | Optional[List[str]] | List of coin names | No |

### Helius API (Solana)

Fetch Solana blockchain data.

```python
from swarms_tools.finance import helius_api_tool

# Fetch account data
account_data = helius_api_tool("account", "account_address")

# Fetch transaction data
tx_data = helius_api_tool("transaction", "tx_signature")

# Fetch token data
token_data = helius_api_tool("token", "token_mint_address")
```

**Arguments:**

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| action | str | Type of action ('account', 'transaction', 'token') | Yes |
| identifier | str | Address/signature to query | Yes |

### DexScreener API

Fetch DEX trading pair data.

```python
from swarms_tools.finance import (
    fetch_dex_screener_profiles,
    fetch_latest_token_boosts,
    fetch_solana_token_pairs
)

# Fetch latest profiles
profiles = fetch_dex_screener_profiles()

# Fetch token boosts
boosts = fetch_latest_token_boosts()

# Fetch Solana pairs
pairs = fetch_solana_token_pairs(["token_address"])
```

### HTX (Huobi) API

Fetch cryptocurrency data from HTX.

```python
from swarms_tools.finance import fetch_htx_data

# Fetch coin data
data = fetch_htx_data("BTC")
```

**Arguments:**

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| coin_name | str | Cryptocurrency symbol | Yes |

### OKX API

Fetch cryptocurrency data from OKX.

```python
from swarms_tools.finance import okx_api_tool

# Fetch single coin
data = okx_api_tool(["BTC-USDT"])

# Fetch multiple coins
data = okx_api_tool(["BTC-USDT", "ETH-USDT"])
```

**Arguments:**

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| coin_symbols | Optional[List[str]] | List of trading pairs | No |

### EODHD Stock News

Fetch stock market news.

```python
from swarms_tools.finance import fetch_stock_news

# Fetch news for a stock
news = fetch_stock_news("AAPL")
```

**Arguments:**

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| stock_name | str | Stock symbol | Yes |

### Jupiter (Solana DEX)

Fetch Solana DEX prices.

```python
from swarms_tools.finance import get_jupiter_price

# Fetch price data
price = get_jupiter_price(input_mint="input_token", output_mint="output_token")
```

**Arguments:**

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| input_mint | str | Input token mint address | Yes |
| output_mint | str | Output token mint address | Yes |

### Sector Analysis

Analyze GICS sector ETFs.

```python
from swarms_tools.finance.sector_analysis import analyze_index_sectors

# Run sector analysis
analyze_index_sectors()
```

### Solana Tools

Check Solana wallet balances and manage tokens.

```python
from swarms_tools.finance import check_solana_balance, check_multiple_wallets

# Check single wallet
balance = check_solana_balance("wallet_address")

# Check multiple wallets
balances = check_multiple_wallets(["wallet1", "wallet2"])
```

**Arguments:**

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| wallet_address | str | Solana wallet address | Yes |
| wallet_addresses | List[str] | List of wallet addresses | Yes |

## Complete Example

Here's a comprehensive example using multiple tools:

```python
from swarms_tools.finance import (
    yahoo_finance_api,
    coin_gecko_coin_api,
    coinmarketcap_api,
    fetch_htx_data
)

# Fetch stock data
stocks = yahoo_finance_api(["AAPL", "GOOG"])
print("Stock Data:", stocks)

# Fetch crypto data from multiple sources
bitcoin_cg = coin_gecko_coin_api("bitcoin")
print("Bitcoin Data (CoinGecko):", bitcoin_cg)

crypto_cmc = coinmarketcap_api(["Bitcoin", "Ethereum"])
print("Crypto Data (CoinMarketCap):", crypto_cmc)

btc_htx = fetch_htx_data("BTC")
print("Bitcoin Data (HTX):", btc_htx)
```

## Error Handling

All tools include proper error handling and logging. Errors are logged using the `loguru` logger. Example error handling:

```python
from loguru import logger

try:
    data = yahoo_finance_api(["INVALID"])
except Exception as e:
    logger.error(f"Error fetching stock data: {e}")
```

## Rate Limits

Please be aware of rate limits for various APIs:
- CoinGecko: 50 calls/minute (free tier)
- CoinMarketCap: Varies by subscription
- Helius: Varies by subscription
- DexScreener: 300 calls/minute for pairs, 60 calls/minute for profiles
- Other APIs: Refer to respective documentation

## Dependencies

The package automatically handles most dependencies, but you may need to install some manually:
