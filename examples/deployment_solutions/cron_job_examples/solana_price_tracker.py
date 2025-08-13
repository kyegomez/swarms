from swarms import Agent, CronJob
from loguru import logger
import requests
import json
from datetime import datetime


def get_solana_price() -> str:
    """
    Fetches comprehensive Solana (SOL) price data using CoinGecko API.

    Returns:
        str: A JSON formatted string containing Solana's current price and market data including:
            - Current price in USD
            - Market cap
            - 24h volume
            - 24h price change
            - Last updated timestamp

    Raises:
        Exception: If there's an error fetching the data from CoinGecko API
    """
    try:
        # CoinGecko API endpoint for simple price data
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": "solana",  # Solana's CoinGecko ID
            "vs_currencies": "usd",
            "include_market_cap": True,
            "include_24hr_vol": True,
            "include_24hr_change": True,
            "include_last_updated_at": True,
        }

        # Make API request with timeout
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        # Parse response data
        data = response.json()

        if "solana" not in data:
            raise Exception("Solana data not found in API response")

        solana_data = data["solana"]

        # Compile comprehensive data
        solana_info = {
            "timestamp": datetime.now().isoformat(),
            "coin_info": {
                "name": "Solana",
                "symbol": "SOL",
                "coin_id": "solana",
            },
            "price_data": {
                "current_price_usd": solana_data.get("usd", "N/A"),
                "market_cap_usd": solana_data.get(
                    "usd_market_cap", "N/A"
                ),
                "volume_24h_usd": solana_data.get(
                    "usd_24h_vol", "N/A"
                ),
                "price_change_24h_percent": solana_data.get(
                    "usd_24h_change", "N/A"
                ),
                "last_updated_at": solana_data.get(
                    "last_updated_at", "N/A"
                ),
            },
            "formatted_data": {
                "price_formatted": (
                    f"${solana_data.get('usd', 'N/A'):,.2f}"
                    if solana_data.get("usd")
                    else "N/A"
                ),
                "market_cap_formatted": (
                    f"${solana_data.get('usd_market_cap', 'N/A'):,.0f}"
                    if solana_data.get("usd_market_cap")
                    else "N/A"
                ),
                "volume_formatted": (
                    f"${solana_data.get('usd_24h_vol', 'N/A'):,.0f}"
                    if solana_data.get("usd_24h_vol")
                    else "N/A"
                ),
                "change_formatted": (
                    f"{solana_data.get('usd_24h_change', 'N/A'):+.2f}%"
                    if solana_data.get("usd_24h_change") is not None
                    else "N/A"
                ),
            },
        }

        logger.info(
            f"Successfully fetched Solana price: ${solana_data.get('usd', 'N/A')}"
        )
        return json.dumps(solana_info, indent=4)

    except requests.RequestException as e:
        error_msg = f"API request failed: {e}"
        logger.error(error_msg)
        return json.dumps(
            {
                "error": error_msg,
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
            },
            indent=4,
        )
    except Exception as e:
        error_msg = f"Error fetching Solana price data: {e}"
        logger.error(error_msg)
        return json.dumps(
            {
                "error": error_msg,
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
            },
            indent=4,
        )


def analyze_solana_data(data: str) -> str:
    """
    Analyzes Solana price data and provides insights.

    Args:
        data (str): JSON string containing Solana price data

    Returns:
        str: Analysis and insights about the current Solana market data
    """
    try:
        # Parse the data
        solana_data = json.loads(data)

        if "error" in solana_data:
            return f"❌ Error in data: {solana_data['error']}"

        price_data = solana_data.get("price_data", {})
        formatted_data = solana_data.get("formatted_data", {})

        # Extract key metrics
        price_data.get("current_price_usd")
        price_change = price_data.get("price_change_24h_percent")
        volume_24h = price_data.get("volume_24h_usd")
        market_cap = price_data.get("market_cap_usd")

        # Generate analysis
        analysis = f"""
🔍 **Solana (SOL) Market Analysis** - {solana_data.get('timestamp', 'N/A')}

💰 **Current Price**: {formatted_data.get('price_formatted', 'N/A')}
📊 **24h Change**: {formatted_data.get('change_formatted', 'N/A')}
💎 **Market Cap**: {formatted_data.get('market_cap_formatted', 'N/A')}
📈 **24h Volume**: {formatted_data.get('volume_formatted', 'N/A')}

"""

        # Add sentiment analysis based on price change
        if price_change is not None:
            if price_change > 5:
                analysis += "🚀 **Sentiment**: Strongly Bullish - Significant positive momentum\n"
            elif price_change > 1:
                analysis += "📈 **Sentiment**: Bullish - Positive price action\n"
            elif price_change > -1:
                analysis += (
                    "➡️ **Sentiment**: Neutral - Sideways movement\n"
                )
            elif price_change > -5:
                analysis += "📉 **Sentiment**: Bearish - Negative price action\n"
            else:
                analysis += "🔻 **Sentiment**: Strongly Bearish - Significant decline\n"

        # Add volume analysis
        if volume_24h and market_cap:
            try:
                volume_market_cap_ratio = (
                    volume_24h / market_cap
                ) * 100
                if volume_market_cap_ratio > 10:
                    analysis += "🔥 **Volume**: High trading activity - Strong market interest\n"
                elif volume_market_cap_ratio > 5:
                    analysis += (
                        "📊 **Volume**: Moderate trading activity\n"
                    )
                else:
                    analysis += "😴 **Volume**: Low trading activity - Limited market movement\n"
            except (TypeError, ZeroDivisionError):
                analysis += "📊 **Volume**: Unable to calculate volume/market cap ratio\n"

        analysis += f"\n⏰ **Last Updated**: {price_data.get('last_updated_at', 'N/A')}"

        return analysis

    except json.JSONDecodeError as e:
        return f"❌ Error parsing data: {e}"
    except Exception as e:
        return f"❌ Error analyzing data: {e}"


# Initialize the Solana analysis agent
agent = Agent(
    agent_name="Solana-Price-Analyzer",
    agent_description="Specialized agent for analyzing Solana (SOL) cryptocurrency price data and market trends",
    system_prompt=f"""You are an expert cryptocurrency analyst specializing in Solana (SOL) analysis. Your expertise includes:

- Technical analysis and chart patterns
- Market sentiment analysis
- Volume and liquidity analysis
- Price action interpretation
- Market cap and valuation metrics
- Cryptocurrency market dynamics
- DeFi ecosystem analysis
- Blockchain technology trends

When analyzing Solana data, you should:
- Evaluate price movements and trends
- Assess market sentiment and momentum
- Consider volume and liquidity factors
- Analyze market cap positioning
- Provide actionable insights
- Identify potential catalysts or risks
- Consider broader market context

You communicate clearly and provide practical analysis that helps users understand Solana's current market position and potential future movements.

Current Solana Data: {get_solana_price()}
""",
    max_loops=1,
    model_name="gpt-4o-mini",
    dynamic_temperature_enabled=True,
    output_type="str-all-except-first",
    streaming_on=False,  # need to fix this bug where streaming is working but makes copies of the border when you scroll on the terminal
    print_on=True,
    telemetry_enable=False,
)


def main():
    """
    Main function to run the Solana price tracking cron job.
    """
    logger.info("🚀 Starting Solana price tracking cron job")
    logger.info("📊 Fetching Solana price every 10 seconds...")

    # Create cron job that runs every 10 seconds
    cron_job = CronJob(agent=agent, interval="30seconds")

    # Run the cron job with analysis task
    cron_job.run(
        task="Analyze the current Solana (SOL) price data comprehensively. Provide detailed market analysis including price trends, volume analysis, market sentiment, and actionable insights. Format your response clearly with emojis and structured sections."
    )


if __name__ == "__main__":
    main()
