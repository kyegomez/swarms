"""
Simple Cryptocurrency Concurrent CronJob Example

This is a simplified version showcasing the core concept of combining:
- CronJob (for scheduling)
- ConcurrentWorkflow (for parallel execution)
- Each agent analyzes a specific cryptocurrency

Perfect for understanding the basic pattern before diving into the full example.
"""

import json
import requests
from datetime import datetime
from loguru import logger

from swarms import Agent, CronJob, ConcurrentWorkflow


def get_specific_crypto_data(coin_ids):
    """Fetch specific crypto data from CoinGecko API."""
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": ",".join(coin_ids),
            "vs_currencies": "usd",
            "include_24hr_change": True,
            "include_market_cap": True,
            "include_24hr_vol": True,
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        result = {
            "timestamp": datetime.now().isoformat(),
            "coins": data,
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error fetching crypto data: {e}")
        return f"Error: {e}"


def create_crypto_specific_agents():
    """Create agents that each specialize in one cryptocurrency."""

    # Bitcoin Specialist Agent
    bitcoin_agent = Agent(
        agent_name="Bitcoin-Analyst",
        system_prompt="""You are a Bitcoin specialist. Analyze ONLY Bitcoin (BTC) data from the provided dataset. 
        Focus on:
        - Bitcoin price movements and trends
        - Market dominance and institutional adoption
        - Bitcoin-specific market dynamics
        - Store of value characteristics
        Ignore all other cryptocurrencies in your analysis.""",
        model_name="gpt-4o-mini",
        max_loops=1,
        print_on=False,  # Important for concurrent execution
    )

    # Ethereum Specialist Agent
    ethereum_agent = Agent(
        agent_name="Ethereum-Analyst",
        system_prompt="""You are an Ethereum specialist. Analyze ONLY Ethereum (ETH) data from the provided dataset.
        Focus on:
        - Ethereum price action and DeFi ecosystem
        - Smart contract platform adoption
        - Gas fees and network usage
        - Layer 2 scaling solutions impact
        Ignore all other cryptocurrencies in your analysis.""",
        model_name="gpt-4o-mini",
        max_loops=1,
        print_on=False,
    )

    # Solana Specialist Agent
    solana_agent = Agent(
        agent_name="Solana-Analyst",
        system_prompt="""You are a Solana specialist. Analyze ONLY Solana (SOL) data from the provided dataset.
        Focus on:
        - Solana price performance and ecosystem growth
        - High-performance blockchain advantages
        - DeFi and NFT activity on Solana
        - Network reliability and uptime
        Ignore all other cryptocurrencies in your analysis.""",
        model_name="gpt-4o-mini",
        max_loops=1,
        print_on=False,
    )

    return [bitcoin_agent, ethereum_agent, solana_agent]


def main():
    """Main function demonstrating crypto-specific concurrent analysis with cron job."""
    logger.info(
        "🚀 Starting Simple Crypto-Specific Concurrent Analysis"
    )
    logger.info("💰 Each agent analyzes one specific cryptocurrency:")
    logger.info("   🟠 Bitcoin-Analyst -> BTC only")
    logger.info("   🔵 Ethereum-Analyst -> ETH only")
    logger.info("   🟢 Solana-Analyst -> SOL only")

    # Define specific cryptocurrencies to analyze
    coin_ids = ["bitcoin", "ethereum", "solana"]

    # Step 1: Create crypto-specific agents
    agents = create_crypto_specific_agents()

    # Step 2: Create ConcurrentWorkflow
    workflow = ConcurrentWorkflow(
        name="Simple-Crypto-Specific-Analysis",
        agents=agents,
        show_dashboard=True,  # Shows real-time progress
    )

    # Step 3: Create CronJob with the workflow
    cron_job = CronJob(
        agent=workflow,  # Use workflow as the agent
        interval="60seconds",  # Run every minute
        job_id="simple-crypto-specific-cron",
    )

    # Step 4: Define the analysis task
    task = f"""
    Analyze the cryptocurrency data below. Each agent should focus ONLY on their assigned cryptocurrency:
    
    - Bitcoin-Analyst: Analyze Bitcoin (BTC) data only
    - Ethereum-Analyst: Analyze Ethereum (ETH) data only
    - Solana-Analyst: Analyze Solana (SOL) data only
    
    Cryptocurrency Data:
    {get_specific_crypto_data(coin_ids)}
    
    Each agent should:
    1. Extract and analyze data for YOUR ASSIGNED cryptocurrency only
    2. Provide brief insights from your specialty perspective
    3. Give a price trend assessment
    4. Identify key opportunities or risks
    5. Ignore all other cryptocurrencies
    """

    # Step 5: Start the cron job
    logger.info("▶️  Starting cron job - Press Ctrl+C to stop")
    try:
        cron_job.run(task=task)
    except KeyboardInterrupt:
        logger.info("⏹️  Stopped by user")


if __name__ == "__main__":
    main()
