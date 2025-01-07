import os
import requests
from dotenv import load_dotenv
from swarm_models import OpenAIChat
from swarms import Agent, GroupChat


load_dotenv()

# Initialize base model
model = OpenAIChat(
    model_name="gpt-4o",
    max_tokens=4000,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# Technical Analysis Agent
TECHNICAL_ANALYST_PROMPT = """
You are an expert Technical Analysis agent specializing in cryptocurrency markets. Your role is to:
1. Analyze price patterns, trends, and indicators
2. Identify key support and resistance levels
3. Evaluate market momentum and volume patterns
4. Provide detailed technical insights based on chart patterns
5. Monitor and interpret trading indicators (RSI, MACD, etc.)

When analyzing data, focus on:
- Price action and volume analysis
- Chart pattern recognition
- Indicator divergences and confluences
- Market structure and trend analysis
- Support/resistance levels and price zones

Provide your analysis in a clear, structured format with bullet points for key findings.
Emphasize objective technical analysis without making direct price predictions.
"""

technical_agent = Agent(
    agent_name="Technical-Analyst",
    system_prompt=TECHNICAL_ANALYST_PROMPT,
    llm=model,
    max_loops=1,
    verbose=True,
    dynamic_temperature_enabled=True,
)

# Fundamental Analysis Agent
FUNDAMENTAL_ANALYST_PROMPT = """
You are a Fundamental Analysis expert focusing on cryptocurrency project evaluation. You analyze:
1. Tokenomics and monetary policy
2. Team composition and development activity
3. Market competition and positioning
4. Partnerships and integrations
5. On-chain metrics and network activity

Key focus areas:
- Token distribution and supply metrics
- Development activity and GitHub commits
- Network growth and adoption metrics
- Competition analysis and market positioning
- Partnership quality and strategic value

Provide comprehensive fundamental analysis based on verifiable data.
Focus on long-term value drivers and project sustainability.
"""

fundamental_agent = Agent(
    agent_name="Fundamental-Analyst",
    system_prompt=FUNDAMENTAL_ANALYST_PROMPT,
    llm=model,
    max_loops=1,
    verbose=True,
    dynamic_temperature_enabled=True,
)

# Market Sentiment Agent
SENTIMENT_ANALYST_PROMPT = """
You are a Market Sentiment Analysis specialist focusing on social and market psychology. Your tasks:
1. Analyze social media sentiment across platforms
2. Monitor community engagement metrics
3. Track market fear/greed indicators
4. Evaluate news impact on market sentiment
5. Assess institutional interest and whale activity

Focus areas:
- Social media sentiment trends
- Community growth and engagement
- News sentiment analysis
- Institutional investment flows
- Whale wallet activity monitoring

Provide objective sentiment analysis using multiple data sources.
Emphasize data-driven insights rather than speculation.
"""

sentiment_agent = Agent(
    agent_name="Sentiment-Analyst",
    system_prompt=SENTIMENT_ANALYST_PROMPT,
    llm=model,
    max_loops=1,
    verbose=True,
    dynamic_temperature_enabled=True,
)

# Risk Management Agent
RISK_MANAGER_PROMPT = """
You are a Risk Management specialist focused on market risk assessment. Your role involves:
1. Identifying potential market risks and vulnerabilities
2. Analyzing market volatility and liquidity
3. Evaluating correlation with broader markets
4. Assessing regulatory and operational risks
5. Monitoring market manipulation indicators

Key focus areas:
- Volatility analysis and risk metrics
- Liquidity depth assessment
- Correlation analysis with major assets
- Regulatory compliance risks
- Smart contract and protocol risks

Provide comprehensive risk assessment with clear risk ratings and mitigation strategies.
Focus on identifying both obvious and subtle risk factors.
"""

risk_agent = Agent(
    agent_name="Risk-Manager",
    system_prompt=RISK_MANAGER_PROMPT,
    llm=model,
    max_loops=1,
    verbose=True,
    dynamic_temperature_enabled=True,
)

# Macro Analysis Agent
MACRO_ANALYST_PROMPT = """
You are a Macro Analysis specialist focusing on broader market context. Your role involves:
1. Analyzing global economic trends
2. Evaluating crypto market cycles
3. Monitoring regulatory developments
4. Assessing cross-market correlations
5. Analyzing institutional trends

Key focus areas:
- Global economic indicators
- Crypto market cycle analysis
- Regulatory landscape changes
- Institutional adoption trends
- Cross-asset correlations

Provide macro context and analysis of how broader trends affect the crypto market.
Focus on identifying major market-moving factors and trends.
"""

macro_agent = Agent(
    agent_name="Macro-Analyst",
    system_prompt=MACRO_ANALYST_PROMPT,
    llm=model,
    max_loops=1,
    verbose=True,
    dynamic_temperature_enabled=True,
)

# Create group chat with all agents
agents = [
    technical_agent,
    fundamental_agent,
    sentiment_agent,
    risk_agent,
    macro_agent,
]


# # Initialize the agent
# swarms_agent = Agent(
#     agent_name="Swarms-Token-Agent",
#     system_prompt=SWARMS_AGENT_SYS_PROMPT,
#     llm=model,
#     max_loops=1,
#     autosave=True,
#     dashboard=False,
#     verbose=True,
#     dynamic_temperature_enabled=True,
#     saved_state_path="swarms_agent.json",
#     user_name="swarms_corp",
#     retry_attempts=1,
#     context_length=200000,
#     return_step_meta=False,
#     output_type="string",
#     streaming_on=False,
# )


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


data = fetch_htx_data("swarms")

swarm = GroupChat(
    name="htx-swarm",
    description="Swarm that analyzes data from HTX",
    agents=agents,
    max_loops=1,
)

out = swarm.run(
    f"Have a internal strategic discussion with the other agents on the price action of the swarms coin {str(data)}"
)
print(out.model_dump_json(indent=4))
