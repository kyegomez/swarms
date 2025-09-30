# Quant Crypto Agent

- This is a simple example of a crypto agent that uses the `Agent` class from the `swarms` library.
- It uses the `fetch_htx_data` and `coin_gecko_coin_api` tools to fetch data from the `htx` and `CoinGecko` APIs.
- It uses the `Agent` class to create an agent that can analyze the current state of a crypto asset.

## Steps

1. Install the `swarms` library.
2. Install the `swarms_tools` library.
3. Setup your `.env` file with the `OPENAI_API_KEY` environment variables.
4. Run the code.

## Installation:

```bash
pip install swarms swarms-tools python-dotenv
```

## Code:

```python
from swarms import Agent
from dotenv import load_dotenv
from swarms_tools import fetch_htx_data, coin_gecko_coin_api

load_dotenv()

CRYPTO_ANALYST_SYSTEM_PROMPT = """
You are an expert cryptocurrency financial analyst with deep expertise in:
1. Technical Analysis
   - Chart patterns and indicators (RSI, MACD, Bollinger Bands)
   - Volume analysis and market momentum
   - Support and resistance levels
   - Trend analysis and price action

2. Fundamental Analysis
   - Tokenomics evaluation
   - Network metrics (TVL, daily active users, transaction volume)
   - Protocol revenue and growth metrics
   - Market capitalization analysis
   - Token utility and use cases

3. Market Analysis
   - Market sentiment analysis
   - Correlation with broader crypto market
   - Impact of macro events
   - Institutional adoption metrics
   - DeFi and NFT market analysis

4. Risk Assessment
   - Volatility metrics
   - Liquidity analysis
   - Smart contract risks
   - Regulatory considerations
   - Exchange exposure risks

5. Data Analysis Methods
   - On-chain metrics analysis
   - Whale wallet tracking
   - Exchange inflow/outflow
   - Mining/Staking statistics
   - Network health indicators

When analyzing crypto assets, always:
1. Start with a comprehensive market overview
2. Examine both on-chain and off-chain metrics
3. Consider multiple timeframes (short, medium, long-term)
4. Evaluate risk-reward ratios
5. Assess market sentiment and momentum
6. Consider regulatory and security factors
7. Analyze correlations with BTC, ETH, and traditional markets
8. Examine liquidity and volume profiles
9. Review recent protocol developments and updates
10. Consider macro economic factors

Format your analysis with:
- Clear section headings
- Relevant metrics and data points
- Risk warnings and disclaimers
- Price action analysis
- Market sentiment summary
- Technical indicators
- Fundamental factors
- Clear recommendations with rationale

Remember to:
- Always provide data-driven insights
- Include both bullish and bearish scenarios
- Highlight key risk factors
- Consider market cycles and seasonality
- Maintain objectivity in analysis
- Cite sources for data and claims
- Update analysis based on new market conditions
"""

# Initialize the crypto analysis agent
agent = Agent(
    agent_name="Crypto-Analysis-Expert",
    agent_description="Expert cryptocurrency financial analyst and market researcher",
    system_prompt=CRYPTO_ANALYST_SYSTEM_PROMPT,
    max_loops="auto",
    model_name="gpt-4.1",
    dynamic_temperature_enabled=True,
    user_name="crypto_analyst",
    output_type="str",
    interactive=True,
)

print(fetch_htx_data("sol"))
print(coin_gecko_coin_api("solana"))

# Example usage
agent.run(
    f"""
    Analyze the current state of Solana (SOL), including:
    1. Technical analysis of price action
    2. On-chain metrics and network health
    3. Recent protocol developments
    4. Market sentiment
    5. Risk factors
    Please provide a comprehensive analysis with data-driven insights.
    
    # Solana CoinGecko Data
    Real-tim data from Solana CoinGecko: \n {coin_gecko_coin_api("solana")}
    
    """
)
```