from swarms import Agent, SwarmRouter

# Agent 1: Risk Metrics Calculator
risk_metrics_agent = Agent(
    agent_name="Risk-Metrics-Calculator",
    agent_description="Calculates key risk metrics like VaR, Sharpe ratio, and volatility",
    system_prompt="""You are a risk metrics specialist. Calculate and explain:
    - Value at Risk (VaR)
    - Sharpe ratio
    - Volatility
    - Maximum drawdown
    - Beta coefficient
    
    Provide clear, numerical results with brief explanations.""",
    max_loops=1,
    model_name="gpt-4.1",
    random_model_enabled=True,
    dynamic_temperature_enabled=True,
    output_type="str-all-except-first",
    max_tokens=4096,
)

# Agent 2: Portfolio Risk Analyzer
portfolio_risk_agent = Agent(
    agent_name="Portfolio-Risk-Analyzer",
    agent_description="Analyzes portfolio diversification and concentration risk",
    system_prompt="""You are a portfolio risk analyst. Focus on:
    - Portfolio diversification analysis
    - Concentration risk assessment
    - Correlation analysis
    - Sector/asset allocation risk
    - Liquidity risk evaluation
    
    Provide actionable insights for risk reduction.""",
    max_loops=1,
    model_name="gpt-4.1",
    random_model_enabled=True,
    dynamic_temperature_enabled=True,
    output_type="str-all-except-first",
    max_tokens=4096,
)

# Agent 3: Market Risk Monitor
market_risk_agent = Agent(
    agent_name="Market-Risk-Monitor",
    agent_description="Monitors market conditions and identifies risk factors",
    system_prompt="""You are a market risk monitor. Identify and assess:
    - Market volatility trends
    - Economic risk factors
    - Geopolitical risks
    - Interest rate risks
    - Currency risks
    
    Provide current risk alerts and trends.""",
    max_loops=1,
    model_name="gpt-4.1",
    random_model_enabled=True,
    dynamic_temperature_enabled=True,
    output_type="str-all-except-first",
    max_tokens=4096,
)


swarm = SwarmRouter(
    name="SwarmRouter",
    description="A router that can route messages to the appropriate swarm",
    agents=[
        risk_metrics_agent,
        portfolio_risk_agent,
    ],
    max_loops=1,
    swarm_type="SequentialWorkflow",
    output_type="final",
)


out = swarm.run(
    "What are the best ways to short the EU markets. Give me specific tickrs to short and strategies to use. Create a comprehensive report with all the information you can find."
)

print(out)
