import requests
from dotenv import load_dotenv
from swarms import Agent, SequentialWorkflow


load_dotenv()


# Technical Analysis Agent
TECHNICAL_ANALYST_PROMPT = """
You are an expert Technical Analysis agent specializing in cryptocurrency markets. Your role is to:
1. Analyze price patterns, trends, and indicators with specific numerical data.
2. Identify key support and resistance levels with exact price points.
3. Evaluate market momentum and volume patterns using quantitative metrics.
4. Provide detailed technical insights based on chart patterns, including Fibonacci retracement levels and moving averages.
5. Monitor and interpret trading indicators (RSI, MACD, Bollinger Bands) with specific values.

When analyzing data, focus on:
- Price action and volume analysis, including percentage changes and volume spikes.
- Chart pattern recognition, such as head and shoulders, double tops/bottoms, and triangles.
- Indicator divergences and confluences, providing specific indicator values.
- Market structure and trend analysis, including bullish/bearish trends with defined price ranges.
- Support/resistance levels and price zones, specifying exact price levels.

Provide your analysis in a clear, structured format with bullet points for key findings, including numerical data.
Emphasize objective technical analysis without making direct price predictions.
"""

technical_agent = Agent(
    agent_name="Technical-Analyst",
    system_prompt=TECHNICAL_ANALYST_PROMPT,
    model_name="gpt-4o",
    max_loops=1,
    verbose=True,
    dynamic_temperature_enabled=True,
)


# Market Sentiment Agent
SENTIMENT_ANALYST_PROMPT = """
You are a Market Sentiment Analysis specialist focusing on social and market psychology. Your tasks:
1. Analyze social media sentiment across platforms with specific sentiment scores.
2. Monitor community engagement metrics, including likes, shares, and comments.
3. Track market fear/greed indicators with numerical values.
4. Evaluate news impact on market sentiment, providing specific examples.
5. Assess institutional interest and whale activity, detailing transaction sizes.

Focus areas:
- Social media sentiment trends, providing percentage changes.
- Community growth and engagement metrics, including follower counts.
- News sentiment analysis, quantifying positive/negative impacts.
- Institutional investment flows, detailing specific amounts.
- Whale wallet activity monitoring, providing transaction details.

Provide objective sentiment analysis using multiple data sources.
Emphasize data-driven insights rather than speculation.
"""

# Risk Management Agent
RISK_MANAGER_PROMPT = """
You are a Risk Management specialist focused on market risk assessment. Your role involves:
1. Identifying potential market risks and vulnerabilities with specific examples.
2. Analyzing market volatility and liquidity using quantitative measures.
3. Evaluating correlation with broader markets, providing correlation coefficients.
4. Assessing regulatory and operational risks, detailing specific regulations.
5. Monitoring market manipulation indicators with defined thresholds.

Key focus areas:
- Volatility analysis and risk metrics, including standard deviation and beta values.
- Liquidity depth assessment, providing order book metrics.
- Correlation analysis with major assets, detailing specific correlations.
- Regulatory compliance risks, specifying relevant regulations.
- Smart contract and protocol risks, detailing potential vulnerabilities.

Provide comprehensive risk assessment with clear risk ratings and mitigation strategies.
Focus on identifying both obvious and subtle risk factors.
"""

risk_agent = Agent(
    agent_name="Risk-Manager",
    system_prompt=RISK_MANAGER_PROMPT,
    model_name="gpt-4o",
    max_loops=1,
    verbose=True,
    dynamic_temperature_enabled=True,
)

# Macro Analysis Agent
MACRO_ANALYST_PROMPT = """
You are a Macro Analysis specialist focusing on broader market context. Your role involves:
1. Analyzing global economic trends with specific indicators.
2. Evaluating crypto market cycles, providing cycle duration and phases.
3. Monitoring regulatory developments, detailing specific changes.
4. Assessing cross-market correlations with numerical data.
5. Analyzing institutional trends, providing investment amounts.

Key focus areas:
- Global economic indicators, including GDP growth rates and inflation.
- Crypto market cycle analysis, detailing historical price movements.
- Regulatory landscape changes, specifying impacts on the market.
- Institutional adoption trends, quantifying investment flows.
- Cross-asset correlations, providing correlation coefficients.

Provide macro context and analysis of how broader trends affect the crypto market.
Focus on identifying major market-moving factors and trends.
"""

macro_agent = Agent(
    agent_name="Macro-Analyst",
    system_prompt=MACRO_ANALYST_PROMPT,
    model_name="gpt-4o",
    max_loops=1,
    verbose=True,
    dynamic_temperature_enabled=True,
)

# Create group chat with all agents
agents = [technical_agent, risk_agent, technical_agent]


def fetch_htx_data(coin_name: str):
    base_url = "https://api.huobi.pro"

    # Fetch market ticker data for the coin
    ticker_endpoint = "/market/detail/merged"
    ticker_params = {
        "symbol": f"{coin_name.lower()}usdt"
    }  # Assuming USDT pairing

    try:
        ticker_response = requests.get(
            base_url + ticker_endpoint, params=ticker_params
        )
        ticker_data = ticker_response.json()

        if ticker_data["status"] != "ok":
            return {
                "error": "Unable to fetch ticker data",
                "details": ticker_data,
            }

        # Fetch order book data for the coin
        order_book_endpoint = "/market/depth"
        order_book_params = {
            "symbol": f"{coin_name.lower()}usdt",
            "type": "step0",
        }

        order_book_response = requests.get(
            base_url + order_book_endpoint, params=order_book_params
        )
        order_book_data = order_book_response.json()

        if order_book_data["status"] != "ok":
            return {
                "error": "Unable to fetch order book data",
                "details": order_book_data,
            }

        # Fetch recent trades for the coin
        trades_endpoint = "/market/history/trade"
        trades_params = {
            "symbol": f"{coin_name.lower()}usdt",
            "size": 200,
        }

        trades_response = requests.get(
            base_url + trades_endpoint, params=trades_params
        )
        trades_data = trades_response.json()

        if trades_data["status"] != "ok":
            return {
                "error": "Unable to fetch trade data",
                "details": trades_data,
            }

        # Fetch Kline (Candlestick) data
        kline_endpoint = "/market/history/kline"
        kline_params = {
            "symbol": f"{coin_name.lower()}usdt",
            "period": "1day",
            "size": 200,
        }

        kline_response = requests.get(
            base_url + kline_endpoint, params=kline_params
        )
        kline_data = kline_response.json()

        if kline_data["status"] != "ok":
            return {
                "error": "Unable to fetch kline data",
                "details": kline_data,
            }

        # Format and prepare data for a single coin
        formatted_data = {
            "coin": coin_name.upper(),
            "ticker": {
                "current_price": ticker_data["tick"].get("close"),
                "high": ticker_data["tick"].get("high"),
                "low": ticker_data["tick"].get("low"),
                "open": ticker_data["tick"].get("open"),
                "volume": ticker_data["tick"].get("vol"),
                "amount": ticker_data["tick"].get("amount"),
                "count": ticker_data["tick"].get("count"),
            },
            "order_book": {
                "bids": [
                    {"price": bid[0], "amount": bid[1]}
                    for bid in order_book_data["tick"].get("bids", [])
                ],
                "asks": [
                    {"price": ask[0], "amount": ask[1]}
                    for ask in order_book_data["tick"].get("asks", [])
                ],
            },
            "recent_trades": [
                {
                    "price": trade["data"][0].get("price"),
                    "amount": trade["data"][0].get("amount"),
                    "direction": trade["data"][0].get("direction"),
                    "trade_id": trade["data"][0].get("id"),
                    "timestamp": trade["data"][0].get("ts"),
                }
                for trade in trades_data.get("data", [])
            ],
            "kline_data": [
                {
                    "timestamp": kline["id"],
                    "open": kline["open"],
                    "close": kline["close"],
                    "high": kline["high"],
                    "low": kline["low"],
                    "volume": kline["vol"],
                    "amount": kline.get("amount"),
                }
                for kline in kline_data.get("data", [])
            ],
        }

        return formatted_data

    except requests.exceptions.RequestException as e:
        return {"error": "HTTP request failed", "details": str(e)}


data = fetch_htx_data("eth")

swarm = SequentialWorkflow(
    name="htx-swarm",
    description="Swarm that analyzes data from HTX",
    agents=agents,
    max_loops=1,
)

out = swarm.run(
    f"Analyze the price action of the swarms coin over the past week. {str(data)} Conduct an analysis of the coin and provide a detailed report."
)

print(out)
