from loguru import logger
import yfinance as yf
import json


def get_figma_stock_data(stock: str) -> str:
    """
    Fetches comprehensive stock data for Figma (FIG) using Yahoo Finance.

    Returns:
        Dict[str, Any]: A dictionary containing comprehensive Figma stock data including:
            - Current price and market data
            - Company information
            - Financial metrics
            - Historical data summary
            - Trading statistics

    Raises:
        Exception: If there's an error fetching the data from Yahoo Finance
    """
    try:
        # Initialize Figma stock ticker
        figma = yf.Ticker(stock)

        # Get current stock info
        info = figma.info

        # Get recent historical data (last 30 days)
        hist = figma.history(period="30d")

        # Get real-time fast info
        fast_info = figma.fast_info

        # Compile comprehensive data
        figma_data = {
            "company_info": {
                "name": info.get("longName", "Figma Inc."),
                "symbol": "FIG",
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "website": info.get("website", "N/A"),
                "description": info.get("longBusinessSummary", "N/A"),
            },
            "current_market_data": {
                "current_price": info.get("currentPrice", "N/A"),
                "previous_close": info.get("previousClose", "N/A"),
                "open": info.get("open", "N/A"),
                "day_low": info.get("dayLow", "N/A"),
                "day_high": info.get("dayHigh", "N/A"),
                "volume": info.get("volume", "N/A"),
                "market_cap": info.get("marketCap", "N/A"),
                "price_change": (
                    info.get("currentPrice", 0)
                    - info.get("previousClose", 0)
                    if info.get("currentPrice")
                    and info.get("previousClose")
                    else "N/A"
                ),
                "price_change_percent": info.get(
                    "regularMarketChangePercent", "N/A"
                ),
            },
            "financial_metrics": {
                "pe_ratio": info.get("trailingPE", "N/A"),
                "forward_pe": info.get("forwardPE", "N/A"),
                "price_to_book": info.get("priceToBook", "N/A"),
                "price_to_sales": info.get(
                    "priceToSalesTrailing12Months", "N/A"
                ),
                "enterprise_value": info.get(
                    "enterpriseValue", "N/A"
                ),
                "beta": info.get("beta", "N/A"),
                "dividend_yield": info.get("dividendYield", "N/A"),
                "payout_ratio": info.get("payoutRatio", "N/A"),
            },
            "trading_statistics": {
                "fifty_day_average": info.get(
                    "fiftyDayAverage", "N/A"
                ),
                "two_hundred_day_average": info.get(
                    "twoHundredDayAverage", "N/A"
                ),
                "fifty_two_week_low": info.get(
                    "fiftyTwoWeekLow", "N/A"
                ),
                "fifty_two_week_high": info.get(
                    "fiftyTwoWeekHigh", "N/A"
                ),
                "shares_outstanding": info.get(
                    "sharesOutstanding", "N/A"
                ),
                "float_shares": info.get("floatShares", "N/A"),
                "shares_short": info.get("sharesShort", "N/A"),
                "short_ratio": info.get("shortRatio", "N/A"),
            },
            "recent_performance": {
                "last_30_days": {
                    "start_price": (
                        hist.iloc[0]["Close"]
                        if not hist.empty
                        else "N/A"
                    ),
                    "end_price": (
                        hist.iloc[-1]["Close"]
                        if not hist.empty
                        else "N/A"
                    ),
                    "total_return": (
                        (
                            hist.iloc[-1]["Close"]
                            - hist.iloc[0]["Close"]
                        )
                        / hist.iloc[0]["Close"]
                        * 100
                        if not hist.empty
                        else "N/A"
                    ),
                    "highest_price": (
                        hist["High"].max()
                        if not hist.empty
                        else "N/A"
                    ),
                    "lowest_price": (
                        hist["Low"].min() if not hist.empty else "N/A"
                    ),
                    "average_volume": (
                        hist["Volume"].mean()
                        if not hist.empty
                        else "N/A"
                    ),
                }
            },
            "real_time_data": {
                "last_price": (
                    fast_info.last_price
                    if hasattr(fast_info, "last_price")
                    else "N/A"
                ),
                "last_volume": (
                    fast_info.last_volume
                    if hasattr(fast_info, "last_volume")
                    else "N/A"
                ),
                "bid": (
                    fast_info.bid
                    if hasattr(fast_info, "bid")
                    else "N/A"
                ),
                "ask": (
                    fast_info.ask
                    if hasattr(fast_info, "ask")
                    else "N/A"
                ),
                "bid_size": (
                    fast_info.bid_size
                    if hasattr(fast_info, "bid_size")
                    else "N/A"
                ),
                "ask_size": (
                    fast_info.ask_size
                    if hasattr(fast_info, "ask_size")
                    else "N/A"
                ),
            },
        }

        logger.info("Successfully fetched Figma (FIG) stock data")
        return json.dumps(figma_data, indent=4)

    except Exception as e:
        logger.error(f"Error fetching Figma stock data: {e}")
        raise Exception(f"Failed to fetch Figma stock data: {e}")


# # Example usage
# # Initialize the quantitative trading agent
# agent = Agent(
#     agent_name="Quantitative-Trading-Agent",
#     agent_description="Advanced quantitative trading and algorithmic analysis agent specializing in stock analysis and trading strategies",
#     system_prompt=f"""You are an expert quantitative trading agent with deep expertise in:
#     - Algorithmic trading strategies and implementation
#     - Statistical arbitrage and market making
#     - Risk management and portfolio optimization
#     - High-frequency trading systems
#     - Market microstructure analysis
#     - Quantitative research methodologies
#     - Financial mathematics and stochastic processes
#     - Machine learning applications in trading
#     - Technical analysis and chart patterns
#     - Fundamental analysis and valuation models
#     - Options trading and derivatives
#     - Market sentiment analysis

#     Your core responsibilities include:
#     1. Developing and backtesting trading strategies
#     2. Analyzing market data and identifying alpha opportunities
#     3. Implementing risk management frameworks
#     4. Optimizing portfolio allocations
#     5. Conducting quantitative research
#     6. Monitoring market microstructure
#     7. Evaluating trading system performance
#     8. Performing comprehensive stock analysis
#     9. Generating trading signals and recommendations
#     10. Risk assessment and position sizing

#     When analyzing stocks, you should:
#     - Evaluate technical indicators and chart patterns
#     - Assess fundamental metrics and valuation ratios
#     - Analyze market sentiment and momentum
#     - Consider macroeconomic factors
#     - Provide risk-adjusted return projections
#     - Suggest optimal entry/exit points
#     - Calculate position sizing recommendations
#     - Identify potential catalysts and risks

#     You maintain strict adherence to:
#     - Mathematical rigor in all analyses
#     - Statistical significance in strategy development
#     - Risk-adjusted return optimization
#     - Market impact minimization
#     - Regulatory compliance
#     - Transaction cost analysis
#     - Performance attribution
#     - Data-driven decision making

#     You communicate in precise, technical terms while maintaining clarity for stakeholders.
#     Data: {get_figma_stock_data('FIG')}

#     """,
#     max_loops=1,
#     model_name="gpt-4o-mini",
#     dynamic_temperature_enabled=True,
#     output_type="str-all-except-first",
#     streaming_on=True,
#     print_on=True,
#     telemetry_enable=False,
# )

# # Example 1: Basic usage with just a task
# logger.info("Starting quantitative analysis cron job for Figma (FIG)")
# cron_job = CronJob(agent=agent, interval="10seconds")
# cron_job.run(
#     task="Analyze the Figma (FIG) stock comprehensively using the available stock data. Provide a detailed quantitative analysis"
# )

print(get_figma_stock_data("FIG"))
