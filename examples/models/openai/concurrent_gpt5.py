from swarms import Agent, ConcurrentWorkflow
from swarms_tools import coin_gecko_coin_api

# Create specialized agents for Solana, Bitcoin, Ethereum, Cardano, and Polkadot analysis using CoinGecko API

market_analyst_solana = Agent(
    agent_name="Market-Trend-Analyst-Solana",
    system_prompt="""You are a market trend analyst specializing in Solana (SOL). 
    Analyze SOL price movements, volume patterns, and market sentiment using real-time data from the CoinGecko API.
    Focus on:
    - Technical indicators and chart patterns for Solana
    - Volume analysis and market depth for SOL
    - Short-term and medium-term trend identification
    - Support and resistance levels

    Always use the CoinGecko API tool to fetch up-to-date Solana market data for your analysis.
    Provide actionable insights based on this data.""",
    model_name="claude-sonnet-4-20250514",
    max_loops=1,
    temperature=0.2,
    tools=[coin_gecko_coin_api],
)

market_analyst_bitcoin = Agent(
    agent_name="Market-Trend-Analyst-Bitcoin",
    system_prompt="""You are a market trend analyst specializing in Bitcoin (BTC). 
    Analyze BTC price movements, volume patterns, and market sentiment using real-time data from the CoinGecko API.
    Focus on:
    - Technical indicators and chart patterns for Bitcoin
    - Volume analysis and market depth for BTC
    - Short-term and medium-term trend identification
    - Support and resistance levels

    Always use the CoinGecko API tool to fetch up-to-date Bitcoin market data for your analysis.
    Provide actionable insights based on this data.""",
    model_name="claude-sonnet-4-20250514",
    max_loops=1,
    temperature=0.2,
    tools=[coin_gecko_coin_api],
)

market_analyst_ethereum = Agent(
    agent_name="Market-Trend-Analyst-Ethereum",
    system_prompt="""You are a market trend analyst specializing in Ethereum (ETH). 
    Analyze ETH price movements, volume patterns, and market sentiment using real-time data from the CoinGecko API.
    Focus on:
    - Technical indicators and chart patterns for Ethereum
    - Volume analysis and market depth for ETH
    - Short-term and medium-term trend identification
    - Support and resistance levels

    Always use the CoinGecko API tool to fetch up-to-date Ethereum market data for your analysis.
    Provide actionable insights based on this data.""",
    model_name="claude-sonnet-4-20250514",
    max_loops=1,
    temperature=0.2,
    tools=[coin_gecko_coin_api],
)

market_analyst_cardano = Agent(
    agent_name="Market-Trend-Analyst-Cardano",
    system_prompt="""You are a market trend analyst specializing in Cardano (ADA). 
    Analyze ADA price movements, volume patterns, and market sentiment using real-time data from the CoinGecko API.
    Focus on:
    - Technical indicators and chart patterns for Cardano
    - Volume analysis and market depth for ADA
    - Short-term and medium-term trend identification
    - Support and resistance levels

    Always use the CoinGecko API tool to fetch up-to-date Cardano market data for your analysis.
    Provide actionable insights based on this data.""",
    model_name="claude-sonnet-4-20250514",
    max_loops=1,
    temperature=0.2,
    tools=[coin_gecko_coin_api],
)

market_analyst_polkadot = Agent(
    agent_name="Market-Trend-Analyst-Polkadot",
    system_prompt="""You are a market trend analyst specializing in Polkadot (DOT). 
    Analyze DOT price movements, volume patterns, and market sentiment using real-time data from the CoinGecko API.
    Focus on:
    - Technical indicators and chart patterns for Polkadot
    - Volume analysis and market depth for DOT
    - Short-term and medium-term trend identification
    - Support and resistance levels

    Always use the CoinGecko API tool to fetch up-to-date Polkadot market data for your analysis.
    Provide actionable insights based on this data.""",
    model_name="claude-sonnet-4-20250514",
    max_loops=1,
    temperature=0.2,
    tools=[coin_gecko_coin_api],
)

# Create concurrent workflow
crypto_analysis_swarm = ConcurrentWorkflow(
    agents=[
        market_analyst_solana,
        market_analyst_bitcoin,
        market_analyst_ethereum,
        market_analyst_cardano,
        market_analyst_polkadot,
    ],
    max_loops=1,
)


crypto_analysis_swarm.run(
    "Analyze your own specified coin and create a comprehensive analysis of the coin"
)
