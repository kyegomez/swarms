import asyncio
from swarms import Agent
from dotenv import load_dotenv
from swarms_tools import coin_gecko_coin_api

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

# Initialize multiple crypto analysis agents with different specialties
technical_analyst = Agent(
    agent_name="Technical-Analyst",
    agent_description="Expert in technical analysis and chart patterns",
    system_prompt=CRYPTO_ANALYST_SYSTEM_PROMPT,
    max_loops=1,
    model_name="gpt-4o",
    dynamic_temperature_enabled=True,
    user_name="tech_analyst",
    output_type="str",
)

# List of coins to analyze
coins = ["solana", "raydium", "aixbt", "jupiter"]

# Dictionary to store analyses
coin_analyses = {}


async def analyze_coin(coin, technical_analyst):
    print(f"\n=== Technical Analysis for {coin.upper()} ===\n")

    # Fetch market data
    gecko_data = coin_gecko_coin_api(coin)

    # Get technical analysis
    analysis = await technical_analyst.arun(
        f"""Analyze {coin}'s technical indicators and price action using this data:
        CoinGecko Data: {gecko_data}
        Focus on:
        - Chart patterns and trends
        - Support/resistance levels
        - Momentum indicators
        - Price targets and risk levels
        - Overall technical strength rating (1-10)
        
        End with a clear technical strength score out of 10.
        """
    )
    return coin, analysis


async def main():
    # Create tasks for concurrent execution
    tasks = [analyze_coin(coin, technical_analyst) for coin in coins]

    # Execute all analyses concurrently
    results = await asyncio.gather(*tasks)

    # Store results in coin_analyses
    for coin, analysis in results:
        coin_analyses[coin] = analysis

    # Have technical analyst compare and recommend best investment
    consensus = await technical_analyst.arun(
        f"""Based on your technical analysis of these coins:
        
        Solana Analysis:
        {coin_analyses['solana']}
        
        Raydium Analysis:
        {coin_analyses['raydium']}
        
        Jupiter Analysis:
        {coin_analyses['jupiter']}
        
        AIXBT Analysis:
        {coin_analyses['aixbt']}
        
        Please:
        1. Rank the coins from strongest to weakest technical setup
        2. Identify which coin has the best risk/reward ratio
        3. Make a clear recommendation on which coin is the best investment opportunity and why
        4. Note any key risks or concerns with the recommended coin
        """
    )
    return consensus


# Run the async main function
consensus = asyncio.run(main())
