import asyncio
import aiohttp
from typing import Dict, List, Optional
from datetime import datetime
from statistics import mean, median

from swarms.structs.agent import Agent

# Define the system prompt specialized for $Swarms
SWARMS_AGENT_SYS_PROMPT = """
Here is the extensive prompt for an agent specializing in $Swarms and its ecosystem economics:

---

### Specialized System Prompt: $Swarms Coin & Ecosystem Economics Expert

You are an advanced financial analysis and ecosystem economics agent, specializing in the $Swarms cryptocurrency. Your purpose is to provide in-depth, accurate, and insightful answers about $Swarms, its role in the AI-powered economy, and its tokenomics. Your knowledge spans all aspects of $Swarms, including its vision, roadmap, network effects, and its transformative potential for decentralized agent interactions.

#### Core Competencies:
1. **Tokenomics Expertise**: Understand and explain the supply-demand dynamics, token utility, and value proposition of $Swarms as the foundation of the agentic economy.
2. **Ecosystem Insights**: Articulate the benefits of $Swarms' agent-centric design, universal currency utility, and its impact on fostering innovation and collaboration.
3. **Roadmap Analysis**: Provide detailed insights into the $Swarms roadmap phases, explaining their significance and economic implications.
4. **Real-Time Data Analysis**: Fetch live data such as price, market cap, volume, and 24-hour changes for $Swarms from CoinGecko or other reliable sources.
5. **Economic Visionary**: Analyze how $Swarms supports the democratization of AI and creates a sustainable framework for AI development.

---

#### Your Mission:
You empower users by explaining how $Swarms revolutionizes the AI economy through decentralized agent interactions, seamless value exchange, and frictionless payments. Help users understand how $Swarms incentivizes developers, democratizes access to AI tools, and builds a thriving interconnected economy of autonomous agents.

---

#### Knowledge Base:

##### Vision:
- **Empowering the Agentic Revolution**: $Swarms is the cornerstone of a decentralized AI economy.
- **Mission**: Revolutionize the AI economy by enabling seamless transactions, rewarding excellence, fostering innovation, and lowering entry barriers for developers.

##### Core Features:
1. **Reward Excellence**: Incentivize developers creating high-performing agents.
2. **Seamless Transactions**: Enable frictionless payments for agentic services.
3. **Foster Innovation**: Encourage collaboration and creativity in AI development.
4. **Sustainable Framework**: Provide scalability for long-term AI ecosystem growth.
5. **Democratize AI**: Lower barriers for users and developers to participate in the AI economy.

##### Why $Swarms?
- **Agent-Centric Design**: Each agent operates with its tokenomics, with $Swarms as the base currency for value exchange.
- **Universal Currency**: A single, unified medium for all agent transactions, reducing complexity.
- **Network Effects**: Growing utility and value as more agents join the $Swarms ecosystem.

##### Roadmap:
1. **Phase 1: Foundation**:
   - Launch $Swarms token.
   - Deploy initial agent creation tools.
   - Establish community governance.
2. **Phase 2: Expansion**:
   - Launch agent marketplace.
   - Enable cross-agent communication.
   - Deploy automated market-making tools.
3. **Phase 3: Integration**:
   - Partner with leading AI platforms.
   - Launch developer incentives.
   - Scale the agent ecosystem globally.
4. **Phase 4: Evolution**:
   - Advanced agent capabilities.
   - Cross-chain integration.
   - Create a global AI marketplace.

##### Ecosystem Benefits:
- **Agent Creation**: Simplified deployment of agents with tokenomics built-in.
- **Universal Currency**: Power all agent interactions with $Swarms.
- **Network Effects**: Thrive in an expanding interconnected agent ecosystem.
- **Secure Trading**: Built on Solana for fast and secure transactions.
- **Instant Settlement**: Lightning-fast transactions with minimal fees.
- **Community Governance**: Decentralized decision-making for the ecosystem.

##### Economic Impact:
- Autonomous agents drive value creation independently.
- Exponential growth potential as network effects amplify adoption.
- Interconnected economy fosters innovation and collaboration.

---

#### How to Answer Queries:
1. Always remain neutral, factual, and comprehensive.
2. Include live data where applicable (e.g., price, market cap, trading volume).
3. Structure responses with clear headings and concise explanations.
4. Use context to explain the relevance of $Swarms to the broader AI economy.

---
---

Leverage your knowledge of $Swarms' vision, roadmap, and economics to provide users with insightful and actionable responses. Aim to be the go-to agent for understanding and utilizing $Swarms in the agentic economy.
"""

# Initialize the agent
swarms_agent = Agent(
    agent_name="Swarms-Token-Agent",
    system_prompt=SWARMS_AGENT_SYS_PROMPT,
    model_name="gpt-4o-mini",
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="swarms_agent.json",
    user_name="swarms_corp",
    retry_attempts=1,
    context_length=200000,
    return_step_meta=False,
    output_type="string",
    streaming_on=False,
)


class MultiExchangeDataFetcher:
    def __init__(self):
        self.base_urls = {
            "coingecko": "https://api.coingecko.com/api/v3",
            "dexscreener": "https://api.dexscreener.com/latest/dex",
            "birdeye": "https://public-api.birdeye.so/public",  # Using Birdeye instead of Jupiter
        }

    async def fetch_data(self, url: str) -> Optional[Dict]:
        """Generic async function to fetch data from APIs with error handling"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        return await response.json()
                    print(
                        f"API returned status {response.status} for {url}"
                    )
                    return None
            except asyncio.TimeoutError:
                print(f"Timeout while fetching from {url}")
                return None
            except Exception as e:
                print(f"Error fetching from {url}: {str(e)}")
                return None

    async def get_coingecko_data(self) -> Optional[Dict]:
        """Fetch $Swarms data from CoinGecko"""
        try:
            url = f"{self.base_urls['coingecko']}/simple/price"
            params = {
                "ids": "swarms",
                "vs_currencies": "usd",
                "include_market_cap": "true",
                "include_24hr_vol": "true",
                "include_24hr_change": "true",
            }
            query = f"{url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
            data = await self.fetch_data(query)
            if data and "swarms" in data:
                return {
                    "price": data["swarms"].get("usd"),
                    "volume24h": data["swarms"].get("usd_24h_vol"),
                    "marketCap": data["swarms"].get("usd_market_cap"),
                }
            return None
        except Exception as e:
            print(f"Error processing CoinGecko data: {str(e)}")
            return None

    async def get_dexscreener_data(self) -> Optional[Dict]:
        """Fetch $Swarms data from DexScreener"""
        try:
            url = (
                f"{self.base_urls['dexscreener']}/pairs/solana/swarms"
            )
            data = await self.fetch_data(url)
            if data and "pairs" in data and len(data["pairs"]) > 0:
                pair = data["pairs"][0]  # Get the first pair
                return {
                    "price": float(pair.get("priceUsd", 0)),
                    "volume24h": float(pair.get("volume24h", 0)),
                    "marketCap": float(pair.get("marketCap", 0)),
                }
            return None
        except Exception as e:
            print(f"Error processing DexScreener data: {str(e)}")
            return None

    async def get_birdeye_data(self) -> Optional[Dict]:
        """Fetch $Swarms data from Birdeye"""
        try:
            # Example Birdeye endpoint - replace ADDRESS with actual Swarms token address
            url = f"{self.base_urls['birdeye']}/token/SWRM2bHQFY5ANXzYGdQ8m9ZRMsqFmsWAadLVvHc2ABJ"
            data = await self.fetch_data(url)
            if data and "data" in data:
                token_data = data["data"]
                return {
                    "price": float(token_data.get("price", 0)),
                    "volume24h": float(
                        token_data.get("volume24h", 0)
                    ),
                    "marketCap": float(
                        token_data.get("marketCap", 0)
                    ),
                }
            return None
        except Exception as e:
            print(f"Error processing Birdeye data: {str(e)}")
            return None

    def aggregate_data(
        self, data_points: List[Optional[Dict]]
    ) -> Dict:
        """Aggregate data from multiple sources with null checking"""
        prices = []
        volumes = []
        market_caps = []

        for data in data_points:
            if data and isinstance(data, dict):
                if data.get("price") is not None:
                    prices.append(float(data["price"]))
                if data.get("volume24h") is not None:
                    volumes.append(float(data["volume24h"]))
                if data.get("marketCap") is not None:
                    market_caps.append(float(data["marketCap"]))

        return {
            "price": {
                "mean": mean(prices) if prices else 0,
                "median": median(prices) if prices else 0,
                "min": min(prices) if prices else 0,
                "max": max(prices) if prices else 0,
                "sources": len(prices),
            },
            "volume_24h": {
                "mean": mean(volumes) if volumes else 0,
                "total": sum(volumes) if volumes else 0,
                "sources": len(volumes),
            },
            "market_cap": {
                "mean": mean(market_caps) if market_caps else 0,
                "median": median(market_caps) if market_caps else 0,
                "sources": len(market_caps),
            },
            "timestamp": datetime.now().isoformat(),
            "sources_total": len(
                [d for d in data_points if d is not None]
            ),
        }


async def get_enhanced_swarms_data():
    fetcher = MultiExchangeDataFetcher()

    # Gather all data concurrently
    tasks = [
        fetcher.get_coingecko_data(),
        fetcher.get_dexscreener_data(),
        fetcher.get_birdeye_data(),
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and None values
    valid_results = [r for r in results if isinstance(r, dict)]

    return fetcher.aggregate_data(valid_results)


async def answer_swarms_query(query: str) -> str:
    try:
        # Fetch enhanced data
        swarms_data = await get_enhanced_swarms_data()

        if swarms_data["sources_total"] == 0:
            return "Unable to fetch current market data from any source. Please try again later."

        # Format the data summary with null checks
        data_summary = (
            f"Aggregated Data (from {swarms_data['sources_total']} sources):\n"
            f"Average Price: ${swarms_data['price']['mean']:.4f}\n"
            f"Price Range: ${swarms_data['price']['min']:.4f} - ${swarms_data['price']['max']:.4f}\n"
            f"24hr Volume (Total): ${swarms_data['volume_24h']['total']:,.2f}\n"
            f"Average Market Cap: ${swarms_data['market_cap']['mean']:,.2f}\n"
            f"Last Updated: {swarms_data['timestamp']}"
        )

        # Update the system prompt with the enhanced data capabilities
        enhanced_prompt = (
            SWARMS_AGENT_SYS_PROMPT
            + f"\n\nReal-Time Multi-Exchange Data:\n{data_summary}"
        )

        # Update the agent with the enhanced prompt
        swarms_agent.update_system_prompt(enhanced_prompt)

        # Run the query
        full_query = (
            f"{query}\n\nCurrent Market Data:\n{data_summary}"
        )
        return swarms_agent.run(full_query)
    except Exception as e:
        print(f"Error in answer_swarms_query: {str(e)}")
        return (
            f"An error occurred while processing your query: {str(e)}"
        )


async def main():
    query = "What is the current market status of $Swarms across different exchanges?"
    response = await answer_swarms_query(query)
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
