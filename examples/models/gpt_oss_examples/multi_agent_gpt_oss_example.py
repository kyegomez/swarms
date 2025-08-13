"""
Cryptocurrency Concurrent Multi-Agent Analysis Example

This example demonstrates how to use ConcurrentWorkflow to create
a powerful cryptocurrency tracking system. Each specialized agent analyzes a
specific cryptocurrency concurrently.

Features:
- ConcurrentWorkflow for parallel agent execution
- Each agent specializes in analyzing one specific cryptocurrency
- Real-time data fetching from CoinGecko API
- Concurrent analysis of multiple cryptocurrencies
- Structured output with professional formatting

Architecture:
ConcurrentWorkflow -> [Bitcoin Agent, Ethereum Agent, Solana Agent, etc.] -> Parallel Analysis
"""

from swarms import Agent
from swarms_tools import coin_gecko_coin_api

# Initialize the agent
agent = Agent(
    agent_name="Quantitative-Trading-Agent",
    agent_description="Advanced quantitative trading and algorithmic analysis agent",
    system_prompt="""You are an expert quantitative trading agent with deep expertise in:
    - Algorithmic trading strategies and implementation
    - Statistical arbitrage and market making
    - Risk management and portfolio optimization
    - High-frequency trading systems
    - Market microstructure analysis
    - Quantitative research methodologies
    - Financial mathematics and stochastic processes
    - Machine learning applications in trading
    
    Your core responsibilities include:
    1. Developing and backtesting trading strategies
    2. Analyzing market data and identifying alpha opportunities
    3. Implementing risk management frameworks
    4. Optimizing portfolio allocations
    5. Conducting quantitative research
    6. Monitoring market microstructure
    7. Evaluating trading system performance
    
    You maintain strict adherence to:
    - Mathematical rigor in all analyses
    - Statistical significance in strategy development
    - Risk-adjusted return optimization
    - Market impact minimization
    - Regulatory compliance
    - Transaction cost analysis
    - Performance attribution
    
    You communicate in precise, technical terms while maintaining clarity for stakeholders.""",
    model_name="groq/openai/gpt-oss-120b",
    dynamic_temperature_enabled=True,
    output_type="str-all-except-first",
    max_loops=1,
    streaming_on=True,
)


def main():
    """
    Performs a comprehensive analysis for a list of cryptocurrencies using the agent.
    For each coin, fetches up-to-date market data and requests the agent to provide
    a detailed, actionable, and insightful report including trends, risks, opportunities,
    and technical/fundamental perspectives.
    """
    # Map coin symbols to their CoinGecko IDs
    coin_mapping = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "ADA": "cardano",
        "BNB": "binancecoin",
        "XRP": "ripple",
    }

    for symbol, coin_id in coin_mapping.items():
        try:
            data = coin_gecko_coin_api(coin_id)
            print(f"Data for {symbol}: {data}")

            prompt = (
                f"You are a quantitative trading expert. "
                f"Given the following up-to-date market data for {symbol}:\n\n"
                f"{data}\n\n"
                f"Please provide a thorough analysis including:\n"
                f"- Current price trends and recent volatility\n"
                f"- Key technical indicators and patterns\n"
                f"- Fundamental factors impacting {symbol}\n"
                f"- Potential trading opportunities and associated risks\n"
                f"- Short-term and long-term outlook\n"
                f"- Any notable news or events affecting {symbol}\n"
                f"Conclude with actionable insights and recommendations for traders and investors."
            )
            out = agent.run(task=prompt)
            print(out)

        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            continue


if __name__ == "__main__":
    main()
