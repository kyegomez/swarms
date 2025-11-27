from swarms import Agent
from swarms.structs.hierarchical_swarm import HierarchicalSwarm

# Initialize specialized financial analysis agents
market_research_agent = Agent(
    agent_name="Market-Research-Specialist",
    agent_description="Expert in market research, trend analysis, and competitive intelligence",
    system_prompt="""You are a senior market research specialist with expertise in:
    - Market trend analysis and forecasting
    - Competitive landscape assessment
    - Consumer behavior analysis
    - Industry report generation
    - Market opportunity identification
    - Risk assessment and mitigation strategies
    
    Your responsibilities include:
    1. Conducting comprehensive market research
    2. Analyzing industry trends and patterns
    3. Identifying market opportunities and threats
    4. Evaluating competitive positioning
    5. Providing actionable market insights
    6. Generating detailed research reports
    
    You provide thorough, data-driven analysis with clear recommendations.
    Always cite sources and provide confidence levels for your assessments.""",
    model_name="claude-3-sonnet-20240229",
    max_loops=1,
    temperature=0.7,
)

financial_analyst_agent = Agent(
    agent_name="Financial-Analysis-Expert",
    agent_description="Specialist in financial statement analysis, valuation, and investment research",
    system_prompt="""You are a senior financial analyst with deep expertise in:
    - Financial statement analysis (income statement, balance sheet, cash flow)
    - Valuation methodologies (DCF, comparable company analysis, precedent transactions)
    - Investment research and due diligence
    - Financial modeling and forecasting
    - Risk assessment and portfolio analysis
    - ESG (Environmental, Social, Governance) analysis
    
    Your core responsibilities include:
    1. Analyzing financial statements and key metrics
    2. Conducting valuation analysis using multiple methodologies
    3. Evaluating investment opportunities and risks
    4. Creating financial models and forecasts
    5. Assessing management quality and corporate governance
    6. Providing investment recommendations with clear rationale
    
    You deliver precise, quantitative analysis with supporting calculations and assumptions.
    Always show your work and provide sensitivity analysis for key assumptions.""",
    model_name="claude-3-sonnet-20240229",
    max_loops=1,
    temperature=0.7,
)

technical_analysis_agent = Agent(
    agent_name="Technical-Analysis-Specialist",
    agent_description="Expert in technical analysis, chart patterns, and trading strategies",
    system_prompt="""You are a senior technical analyst with expertise in:
    - Chart pattern recognition and analysis
    - Technical indicators and oscillators
    - Support and resistance level identification
    - Volume analysis and market microstructure
    - Momentum and trend analysis
    - Risk management and position sizing
    
    Your key responsibilities include:
    1. Analyzing price charts and identifying patterns
    2. Evaluating technical indicators and signals
    3. Determining support and resistance levels
    4. Assessing market momentum and trend strength
    5. Providing entry and exit recommendations
    6. Developing risk management strategies
    
    You provide clear, actionable technical analysis with specific price targets and risk levels.
    Always include timeframes and probability assessments for your predictions.""",
    model_name="claude-3-sonnet-20240229",
    max_loops=1,
    temperature=0.7,
)

risk_management_agent = Agent(
    agent_name="Risk-Management-Specialist",
    agent_description="Expert in risk assessment, portfolio management, and regulatory compliance",
    system_prompt="""You are a senior risk management specialist with expertise in:
    - Market risk assessment and measurement
    - Credit risk analysis and evaluation
    - Operational risk identification and mitigation
    - Regulatory compliance and reporting
    - Portfolio optimization and diversification
    - Stress testing and scenario analysis
    
    Your primary responsibilities include:
    1. Identifying and assessing various risk factors
    2. Developing risk mitigation strategies
    3. Conducting stress tests and scenario analysis
    4. Ensuring regulatory compliance
    5. Optimizing risk-adjusted returns
    6. Providing risk management recommendations
    
    You deliver comprehensive risk assessments with quantitative metrics and mitigation strategies.
    Always provide both qualitative and quantitative risk measures with clear action items.""",
    model_name="claude-3-sonnet-20240229",
    max_loops=1,
    temperature=0.7,
)

# Initialize the director agent
director_agent = Agent(
    agent_name="Financial-Analysis-Director",
    agent_description="Senior director who orchestrates comprehensive financial analysis across multiple domains",
    system_prompt="""You are a senior financial analysis director responsible for orchestrating comprehensive 
    financial analysis projects. You coordinate a team of specialized analysts including:
    - Market Research Specialists
    - Financial Analysis Experts  
    - Technical Analysis Specialists
    - Risk Management Specialists
    
    Your role is to:
    1. Break down complex financial analysis tasks into specific, actionable assignments
    2. Assign tasks to the most appropriate specialist based on their expertise
    3. Ensure comprehensive coverage of all analysis dimensions
    4. Coordinate between specialists to avoid duplication and ensure synergy
    5. Synthesize findings from multiple specialists into coherent recommendations
    6. Ensure all analysis meets professional standards and regulatory requirements
    
    You create detailed, specific task assignments that leverage each specialist's unique expertise
    while ensuring the overall analysis is comprehensive and actionable.""",
    model_name="gpt-4o-mini",
    max_loops=1,
    temperature=0.7,
)

# Create list of specialized agents
specialized_agents = [
    market_research_agent,
    financial_analyst_agent,
]

# Initialize the hierarchical swarm
financial_analysis_swarm = HierarchicalSwarm(
    name="Financial-Analysis-Hierarchical-Swarm",
    description="A hierarchical swarm for comprehensive financial analysis with specialized agents coordinated by a director",
    # director=director_agent,
    agents=specialized_agents,
    max_loops=2,
    verbose=True,
)

# Example usage
if __name__ == "__main__":
    # Complex financial analysis task
    task = "Call the Financial-Analysis-Director agent and ask him to analyze the market for Tesla (TSLA)"
    result = financial_analysis_swarm.run(task=task)
    print(result)
