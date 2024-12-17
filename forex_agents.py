from typing import Dict, List
from datetime import datetime
from loguru import logger
from swarms.structs.tree_swarm import TreeAgent, Tree, ForestSwarm
import asyncio
import json
import aiohttp
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

# Configure logging
logger.add("forex_forest.log", rotation="500 MB", level="INFO")


class ForexDataFeed:
    """Real-time forex data collector using free open sources"""

    def __init__(self):
        self.pairs = [
            "EUR/USD",
            "GBP/USD",
            "USD/JPY",
            "AUD/USD",
            "USD/CAD",
        ]

    async def fetch_ecb_rates(self) -> Dict:
        """Fetch exchange rates from European Central Bank (no key required)"""
        try:
            url = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    xml_data = await response.text()

            root = ET.fromstring(xml_data)
            rates = {}
            for cube in root.findall(".//*[@currency]"):
                currency = cube.get("currency")
                rate = float(cube.get("rate"))
                rates[currency] = rate

            # Calculate cross rates
            rates["EUR"] = 1.0  # Base currency
            cross_rates = {}
            for pair in self.pairs:
                base, quote = pair.split("/")
                if base in rates and quote in rates:
                    cross_rates[pair] = rates[base] / rates[quote]

            return cross_rates
        except Exception as e:
            logger.error(f"Error fetching ECB rates: {e}")
            return {}

    async def fetch_forex_factory_data(self) -> Dict:
        """Scrape trading data from Forex Factory"""
        try:
            url = "https://www.forexfactory.com"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=headers
                ) as response:
                    text = await response.text()

            soup = BeautifulSoup(text, "html.parser")

            # Get calendar events
            calendar = []
            calendar_table = soup.find(
                "table", class_="calendar__table"
            )
            if calendar_table:
                for row in calendar_table.find_all(
                    "tr", class_="calendar__row"
                ):
                    try:
                        event = {
                            "currency": row.find(
                                "td", class_="calendar__currency"
                            ).text.strip(),
                            "event": row.find(
                                "td", class_="calendar__event"
                            ).text.strip(),
                            "impact": row.find(
                                "td", class_="calendar__impact"
                            ).text.strip(),
                            "time": row.find(
                                "td", class_="calendar__time"
                            ).text.strip(),
                        }
                        calendar.append(event)
                    except:
                        continue

            return {"calendar": calendar}
        except Exception as e:
            logger.error(f"Error fetching Forex Factory data: {e}")
            return {}

    async def fetch_tradingeconomics_data(self) -> Dict:
        """Scrape economic data from Trading Economics"""
        try:
            url = "https://tradingeconomics.com/calendar"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=headers
                ) as response:
                    text = await response.text()

            soup = BeautifulSoup(text, "html.parser")

            # Get economic indicators
            indicators = []
            calendar_table = soup.find("table", class_="table")
            if calendar_table:
                for row in calendar_table.find_all("tr")[
                    1:
                ]:  # Skip header
                    try:
                        cols = row.find_all("td")
                        indicator = {
                            "country": cols[0].text.strip(),
                            "indicator": cols[1].text.strip(),
                            "actual": cols[2].text.strip(),
                            "previous": cols[3].text.strip(),
                            "consensus": cols[4].text.strip(),
                        }
                        indicators.append(indicator)
                    except:
                        continue

            return {"indicators": indicators}
        except Exception as e:
            logger.error(
                f"Error fetching Trading Economics data: {e}"
            )
            return {}

    async def fetch_dailyfx_data(self) -> Dict:
        """Scrape market analysis from DailyFX"""
        try:
            url = "https://www.dailyfx.com/market-news"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=headers
                ) as response:
                    text = await response.text()

            soup = BeautifulSoup(text, "html.parser")

            # Get market news and analysis
            news = []
            articles = soup.find_all("article", class_="dfx-article")
            for article in articles[:10]:  # Get latest 10 articles
                try:
                    news_item = {
                        "title": article.find("h3").text.strip(),
                        "summary": article.find("p").text.strip(),
                        "currency": article.get(
                            "data-currency", "General"
                        ),
                        "timestamp": article.find("time").get(
                            "datetime"
                        ),
                    }
                    news.append(news_item)
                except:
                    continue

            return {"news": news}
        except Exception as e:
            logger.error(f"Error fetching DailyFX data: {e}")
            return {}

    async def fetch_all_data(self) -> Dict:
        """Fetch and combine all forex data sources"""
        try:
            # Fetch data from all sources concurrently
            rates, ff_data, te_data, dx_data = await asyncio.gather(
                self.fetch_ecb_rates(),
                self.fetch_forex_factory_data(),
                self.fetch_tradingeconomics_data(),
                self.fetch_dailyfx_data(),
            )

            # Combine all data
            market_data = {
                "exchange_rates": rates,
                "calendar": ff_data.get("calendar", []),
                "economic_indicators": te_data.get("indicators", []),
                "market_news": dx_data.get("news", []),
                "timestamp": datetime.now().isoformat(),
            }

            return market_data

        except Exception as e:
            logger.error(f"Error fetching all data: {e}")
            return {}


# Rest of the ForexForestSystem class remains the same...

# (Previous ForexDataFeed class code remains the same...)

# Specialized Agent Prompts
TECHNICAL_ANALYST_PROMPT = """You are an expert forex technical analyst agent.
Your responsibilities:
1. Analyze real-time exchange rate data for patterns and trends
2. Calculate cross-rates and currency correlations
3. Generate trading signals based on price action
4. Monitor market volatility and momentum
5. Identify key support and resistance levels

Data Format:
- You will receive exchange rates from ECB and calculated cross-rates
- Focus on major currency pairs and their relationships
- Consider market volatility and trading volumes

Output Format:
{
    "analysis_type": "technical",
    "timestamp": "ISO timestamp",
    "signals": [
        {
            "pair": "Currency pair",
            "trend": "bullish/bearish/neutral",
            "strength": 1-10,
            "key_levels": {"support": [], "resistance": []},
            "recommendation": "buy/sell/hold"
        }
    ]
}"""

FUNDAMENTAL_ANALYST_PROMPT = """You are an expert forex fundamental analyst agent.
Your responsibilities:
1. Analyze economic calendar events and their impact
2. Evaluate economic indicators from Trading Economics
3. Assess market news and sentiment from DailyFX
4. Monitor central bank actions and policies
5. Track geopolitical events affecting currencies

Data Format:
- Economic calendar events with impact levels
- Latest economic indicators and previous values
- Market news and analysis from reliable sources
- Central bank statements and policy changes

Output Format:
{
    "analysis_type": "fundamental",
    "timestamp": "ISO timestamp",
    "assessments": [
        {
            "currency": "Currency code",
            "economic_outlook": "positive/negative/neutral",
            "key_events": [],
            "impact_score": 1-10,
            "bias": "bullish/bearish/neutral"
        }
    ]
}"""

MARKET_SENTIMENT_PROMPT = """You are an expert market sentiment analysis agent.
Your responsibilities:
1. Analyze news sentiment from DailyFX articles
2. Track market positioning and bias
3. Monitor risk sentiment and market fear/greed
4. Identify potential market drivers
5. Detect sentiment shifts and extremes

Data Format:
- Market news and analysis articles
- Trading sentiment indicators
- Risk event calendar
- Market commentary and analysis

Output Format:
{
    "analysis_type": "sentiment",
    "timestamp": "ISO timestamp",
    "sentiment_data": [
        {
            "pair": "Currency pair",
            "sentiment": "risk-on/risk-off",
            "strength": 1-10,
            "key_drivers": [],
            "outlook": "positive/negative/neutral"
        }
    ]
}"""

STRATEGY_COORDINATOR_PROMPT = """You are the lead forex strategy coordination agent.
Your responsibilities:
1. Synthesize technical, fundamental, and sentiment analysis
2. Generate final trading recommendations
3. Manage risk exposure and position sizing
4. Coordinate entry and exit points
5. Monitor open positions and adjust strategies

Data Format:
- Analysis from technical, fundamental, and sentiment agents
- Current market rates and conditions
- Economic calendar and news events
- Risk parameters and exposure limits

Output Format:
{
    "analysis_type": "strategy",
    "timestamp": "ISO timestamp",
    "recommendations": [
        {
            "pair": "Currency pair",
            "action": "buy/sell/hold",
            "confidence": 1-10,
            "entry_points": [],
            "stop_loss": float,
            "take_profit": float,
            "rationale": "string"
        }
    ]
}"""


class ForexForestSystem:
    """Main system coordinating the forest swarm and data feeds"""

    def __init__(self):
        """Initialize the forex forest system"""
        self.data_feed = ForexDataFeed()

        # Create Technical Analysis Tree
        technical_agents = [
            TreeAgent(
                system_prompt=TECHNICAL_ANALYST_PROMPT,
                agent_name="Price Action Analyst",
                model_name="gpt-4o",
            ),
            TreeAgent(
                system_prompt=TECHNICAL_ANALYST_PROMPT,
                agent_name="Cross Rate Analyst",
                model_name="gpt-4o",
            ),
            TreeAgent(
                system_prompt=TECHNICAL_ANALYST_PROMPT,
                agent_name="Volatility Analyst",
                model_name="gpt-4o",
            ),
        ]

        # Create Fundamental Analysis Tree
        fundamental_agents = [
            TreeAgent(
                system_prompt=FUNDAMENTAL_ANALYST_PROMPT,
                agent_name="Economic Data Analyst",
                model_name="gpt-4o",
            ),
            TreeAgent(
                system_prompt=FUNDAMENTAL_ANALYST_PROMPT,
                agent_name="News Impact Analyst",
                model_name="gpt-4o",
            ),
            TreeAgent(
                system_prompt=FUNDAMENTAL_ANALYST_PROMPT,
                agent_name="Central Bank Analyst",
                model_name="gpt-4o",
            ),
        ]

        # Create Sentiment Analysis Tree
        sentiment_agents = [
            TreeAgent(
                system_prompt=MARKET_SENTIMENT_PROMPT,
                agent_name="News Sentiment Analyst",
                model_name="gpt-4o",
            ),
            TreeAgent(
                system_prompt=MARKET_SENTIMENT_PROMPT,
                agent_name="Risk Sentiment Analyst",
                model_name="gpt-4o",
            ),
            TreeAgent(
                system_prompt=MARKET_SENTIMENT_PROMPT,
                agent_name="Market Positioning Analyst",
                model_name="gpt-4o",
            ),
        ]

        # Create Strategy Coordination Tree
        strategy_agents = [
            TreeAgent(
                system_prompt=STRATEGY_COORDINATOR_PROMPT,
                agent_name="Lead Strategy Coordinator",
                model_name="gpt-4",
                temperature=0.5,
            ),
            TreeAgent(
                system_prompt=STRATEGY_COORDINATOR_PROMPT,
                agent_name="Risk Manager",
                model_name="gpt-4",
                temperature=0.5,
            ),
            TreeAgent(
                system_prompt=STRATEGY_COORDINATOR_PROMPT,
                agent_name="Position Manager",
                model_name="gpt-4",
                temperature=0.5,
            ),
        ]

        # Create trees
        self.technical_tree = Tree(
            tree_name="Technical Analysis", agents=technical_agents
        )
        self.fundamental_tree = Tree(
            tree_name="Fundamental Analysis",
            agents=fundamental_agents,
        )
        self.sentiment_tree = Tree(
            tree_name="Sentiment Analysis", agents=sentiment_agents
        )
        self.strategy_tree = Tree(
            tree_name="Strategy Coordination", agents=strategy_agents
        )

        # Create forest swarm
        self.forest = ForestSwarm(
            trees=[
                self.technical_tree,
                self.fundamental_tree,
                self.sentiment_tree,
                self.strategy_tree,
            ]
        )

        logger.info("Forex Forest System initialized successfully")

    async def prepare_analysis_task(self) -> str:
        """Prepare the analysis task with real-time data"""
        try:
            market_data = await self.data_feed.fetch_all_data()

            task = {
                "action": "analyze_forex_markets",
                "market_data": market_data,
                "timestamp": datetime.now().isoformat(),
                "analysis_required": [
                    "technical",
                    "fundamental",
                    "sentiment",
                    "strategy",
                ],
            }

            return json.dumps(task, indent=2)

        except Exception as e:
            logger.error(f"Error preparing analysis task: {e}")
            raise

    async def run_analysis_cycle(self) -> Dict:
        """Run a complete analysis cycle with the forest swarm"""
        try:
            # Prepare task with real-time data
            task = await self.prepare_analysis_task()

            # Run forest swarm analysis
            result = self.forest.run(task)

            # Parse and validate results
            analysis = (
                json.loads(result)
                if isinstance(result, str)
                else result
            )

            logger.info("Analysis cycle completed successfully")
            return analysis

        except Exception as e:
            logger.error(f"Error in analysis cycle: {e}")
            raise

    async def monitor_markets(self, interval_seconds: int = 300):
        """Continuously monitor markets and run analysis"""
        while True:
            try:
                # Run analysis cycle
                analysis = await self.run_analysis_cycle()

                # Log results
                logger.info("Market analysis completed")
                logger.debug(
                    f"Analysis results: {json.dumps(analysis, indent=2)}"
                )

                # Process any trading signals
                if "recommendations" in analysis:
                    await self.process_trading_signals(
                        analysis["recommendations"]
                    )

                # Wait for next interval
                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Error in market monitoring: {e}")
                await asyncio.sleep(60)

    async def process_trading_signals(
        self, recommendations: List[Dict]
    ):
        """Process and log trading signals from analysis"""
        try:
            for rec in recommendations:
                logger.info(
                    f"Trading Signal: {rec['pair']} - {rec['action']}"
                )
                logger.info(f"Confidence: {rec['confidence']}/10")
                logger.info(f"Entry Points: {rec['entry_points']}")
                logger.info(f"Stop Loss: {rec['stop_loss']}")
                logger.info(f"Take Profit: {rec['take_profit']}")
                logger.info(f"Rationale: {rec['rationale']}")
                logger.info("-" * 50)

        except Exception as e:
            logger.error(f"Error processing trading signals: {e}")


# Example usage
async def main():
    """Main function to run the Forex Forest System"""
    try:
        system = ForexForestSystem()
        await system.monitor_markets()
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    # Set up asyncio event loop and run the system
    asyncio.run(main())
