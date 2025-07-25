from swarms import Agent, ConcurrentWorkflow, SwarmRouter

# Initialize market research agent
market_researcher = Agent(
    agent_name="Market-Researcher",
    system_prompt="""You are a market research specialist. Your tasks include:
    1. Analyzing market trends and patterns
    2. Identifying market opportunities and threats
    3. Evaluating competitor strategies
    4. Assessing customer needs and preferences
    5. Providing actionable market insights""",
    model_name="claude-3-5-sonnet-20240620",
    max_loops=1,
    streaming_on=True,
    print_on=False,
)

# Initialize financial analyst agent
financial_analyst = Agent(
    agent_name="Financial-Analyst",
    system_prompt="""You are a financial analysis expert. Your responsibilities include:
    1. Analyzing financial statements
    2. Evaluating investment opportunities
    3. Assessing risk factors
    4. Providing financial forecasts
    5. Recommending financial strategies""",
    model_name="claude-3-5-sonnet-20240620",
    max_loops=1,
    streaming_on=True,
    print_on=False,
)

# Initialize technical analyst agent
technical_analyst = Agent(
    agent_name="Technical-Analyst",
    system_prompt="""You are a technical analysis specialist. Your focus areas include:
    1. Analyzing price patterns and trends
    2. Evaluating technical indicators
    3. Identifying support and resistance levels
    4. Assessing market momentum
    5. Providing trading recommendations""",
    model_name="claude-3-5-sonnet-20240620",
    max_loops=1,
    streaming_on=True,
    print_on=False,
)

# Create list of agents
agents = [market_researcher, financial_analyst, technical_analyst]

# Initialize the concurrent workflow
workflow = ConcurrentWorkflow(
    name="market-analysis-workflow",
    agents=agents,
    max_loops=1,
    show_dashboard=True,
)

# Run the workflow
result = workflow.run(
    "Analyze Tesla (TSLA) stock from market, financial, and technical perspectives"
)