from swarms import Agent, AgentRearrange

# Create specialized quantitative research agents
weather_data_agent = Agent(
    agent_name="Weather-Data-Agent",
    agent_description="Expert in weather data collection, agricultural commodity research, and meteorological analysis",
    model_name="claude-sonnet-4-20250514",
    max_loops=1,
    system_prompt="""You are a quantitative weather data research specialist. Your role is to:
    1. Collect and analyze weather data from multiple sources (NOAA, Weather APIs, satellite data)
    2. Research agricultural commodity markets and their weather dependencies
    3. Identify weather patterns that historically impact crop yields and commodity prices
    4. Gather data on seasonal weather trends, precipitation patterns, temperature anomalies
    5. Research specific regions and their agricultural production cycles
    6. Collect data on extreme weather events and their market impact
    7. Analyze historical correlations between weather data and commodity price movements
    
    Focus on actionable weather intelligence for trading opportunities. Always provide specific data points, 
    timeframes, and geographic regions. Include confidence levels and data quality assessments.""",
)

quant_analysis_agent = Agent(
    agent_name="Quant-Analysis-Agent",
    agent_description="Expert in quantitative analysis of weather patterns, arbitrage opportunities, and statistical modeling",
    model_name="claude-sonnet-4-20250514",
    max_loops=1,
    system_prompt="""You are a quantitative analysis specialist focused on weather-driven arbitrage opportunities. Your role is to:
    1. Analyze weather data correlations with commodity price movements
    2. Identify statistical arbitrage opportunities in agricultural futures markets
    3. Calculate risk-adjusted returns for weather-based trading strategies
    4. Model price impact scenarios based on weather forecasts
    5. Identify seasonal patterns and mean reversion opportunities
    6. Analyze basis risk and correlation breakdowns between weather and prices
    7. Calculate optimal position sizes and hedging ratios
    8. Assess market inefficiencies in weather-sensitive commodities
    
    Focus on actionable trading signals with specific entry/exit criteria, risk metrics, and expected returns. 
    Always provide quantitative justification and statistical confidence levels.""",
)

trading_strategy_agent = Agent(
    agent_name="Trading-Strategy-Agent",
    agent_description="Expert in trading strategy development, risk assessment, and portfolio management for weather-driven arbitrage",
    model_name="claude-sonnet-4-20250514",
    max_loops=1,
    system_prompt="""You are a quantitative trading strategy specialist focused on weather-driven arbitrage opportunities. Your role is to:
    1. Develop comprehensive trading strategies based on weather data and commodity analysis
    2. Create detailed risk management frameworks for weather-sensitive positions
    3. Design portfolio allocation strategies for agricultural commodity arbitrage
    4. Develop hedging strategies to mitigate weather-related risks
    5. Create position sizing models based on volatility and correlation analysis
    6. Design entry and exit criteria for weather-based trades
    7. Develop contingency plans for unexpected weather events
    8. Create performance monitoring and evaluation frameworks
    
    Focus on practical, implementable trading strategies with clear risk parameters, 
    position management rules, and performance metrics. Always include specific trade setups, 
    risk limits, and monitoring protocols.""",
)

rearrange_system = AgentRearrange(
    agents=[
        weather_data_agent,
        quant_analysis_agent,
        trading_strategy_agent,
    ],
    flow=f"{trading_strategy_agent.agent_name} -> {quant_analysis_agent.agent_name}, {weather_data_agent.agent_name}",
    max_loops=1,
)

rearrange_system.run(
    "What are the best weather trades for the rest of the year 2025? Can we short wheat futures, corn futures, soybean futures, etc.?"
)
