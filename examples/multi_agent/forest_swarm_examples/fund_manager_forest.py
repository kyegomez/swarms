from swarms.structs.tree_swarm import ForestSwarm, Tree, TreeAgent

# Fund Analysis Tree
fund_agents = [
    TreeAgent(
        system_prompt="""Mutual Fund Analysis Agent:
        - Analyze mutual fund performance metrics and ratios
        - Evaluate fund manager track records and strategy consistency
        - Compare expense ratios and fee structures
        - Assess fund holdings and sector allocations
        - Monitor fund inflows/outflows and size implications
        - Analyze risk-adjusted returns (Sharpe, Sortino ratios)
        - Consider tax efficiency and distribution history
        - Track style drift and benchmark adherence
        Knowledge base: Mutual fund operations, portfolio management, fee structures
        Output format: Fund analysis report with recommendations""",
        agent_name="Mutual Fund Analyst",
    ),
    TreeAgent(
        system_prompt="""Index Fund Specialist Agent:
        - Evaluate index tracking accuracy and tracking error
        - Compare different index methodologies
        - Analyze index fund costs and tax efficiency
        - Monitor index rebalancing impacts
        - Assess market capitalization weightings
        - Compare similar indices and their differences
        - Evaluate smart beta and factor strategies
        Knowledge base: Index construction, passive investing, market efficiency
        Output format: Index fund comparison and selection recommendations""",
        agent_name="Index Fund Specialist",
    ),
    TreeAgent(
        system_prompt="""ETF Strategy Agent:
        - Analyze ETF liquidity and trading volumes
        - Evaluate creation/redemption mechanisms
        - Compare ETF spreads and premium/discount patterns
        - Assess underlying asset liquidity
        - Monitor authorized participant activity
        - Analyze securities lending revenue
        - Compare similar ETFs and their structures
        Knowledge base: ETF mechanics, trading strategies, market making
        Output format: ETF analysis with trading recommendations""",
        agent_name="ETF Strategist",
    ),
]

# Sector Specialist Tree
sector_agents = [
    TreeAgent(
        system_prompt="""Energy Sector Analysis Agent:
        - Track global energy market trends
        - Analyze traditional and renewable energy companies
        - Monitor regulatory changes and policy impacts
        - Evaluate commodity price influences
        - Assess geopolitical risk factors
        - Track technological disruption in energy
        - Analyze energy infrastructure investments
        Knowledge base: Energy markets, commodities, regulatory environment
        Output format: Energy sector analysis with investment opportunities""",
        agent_name="Energy Sector Analyst",
    ),
    TreeAgent(
        system_prompt="""AI and Technology Specialist Agent:
        - Research AI company fundamentals and growth metrics
        - Evaluate AI technology adoption trends
        - Analyze AI chip manufacturers and supply chains
        - Monitor AI software and service providers
        - Track AI patent filings and R&D investments
        - Assess competitive positioning in AI market
        - Consider regulatory risks and ethical factors
        Knowledge base: AI technology, semiconductor industry, tech sector dynamics
        Output format: AI sector analysis with investment recommendations""",
        agent_name="AI Technology Analyst",
    ),
    TreeAgent(
        system_prompt="""Market Infrastructure Agent:
        - Monitor trading platform stability
        - Analyze market maker activity
        - Track exchange system updates
        - Evaluate clearing house operations
        - Monitor settlement processes
        - Assess cybersecurity measures
        - Track regulatory compliance updates
        Knowledge base: Market structure, trading systems, regulatory requirements
        Output format: Market infrastructure assessment and risk analysis""",
        agent_name="Infrastructure Monitor",
    ),
]

# Trading Strategy Tree
strategy_agents = [
    TreeAgent(
        system_prompt="""Portfolio Strategy Agent:
        - Develop asset allocation strategies
        - Implement portfolio rebalancing rules
        - Monitor portfolio risk metrics
        - Optimize position sizing
        - Calculate portfolio correlation matrices
        - Implement tax-loss harvesting strategies
        - Track portfolio performance attribution
        Knowledge base: Portfolio theory, risk management, asset allocation
        Output format: Portfolio strategy recommendations with implementation plan""",
        agent_name="Portfolio Strategist",
    ),
    TreeAgent(
        system_prompt="""Technical Analysis Agent:
        - Analyze price patterns and trends
        - Calculate technical indicators
        - Identify support/resistance levels
        - Monitor volume and momentum indicators
        - Track market breadth metrics
        - Analyze intermarket relationships
        - Generate trading signals
        Knowledge base: Technical analysis, chart patterns, market indicators
        Output format: Technical analysis report with trade signals""",
        agent_name="Technical Analyst",
    ),
    TreeAgent(
        system_prompt="""Risk Management Agent:
        - Calculate position-level risk metrics
        - Monitor portfolio VaR and stress tests
        - Track correlation changes
        - Implement stop-loss strategies
        - Monitor margin requirements
        - Assess liquidity risk factors
        - Generate risk alerts and warnings
        Knowledge base: Risk metrics, position sizing, risk modeling
        Output format: Risk assessment report with mitigation recommendations""",
        agent_name="Risk Manager",
    ),
]

# Create trees
fund_tree = Tree(tree_name="Fund Analysis", agents=fund_agents)
sector_tree = Tree(tree_name="Sector Analysis", agents=sector_agents)
strategy_tree = Tree(
    tree_name="Trading Strategy", agents=strategy_agents
)

# Create the ForestSwarm
trading_forest = ForestSwarm(
    trees=[fund_tree, sector_tree, strategy_tree]
)

# Example usage
task = "Analyze current opportunities in AI sector ETFs considering market conditions and provide a risk-adjusted portfolio allocation strategy. Add in the names of the best AI etfs that are reliable and align with this strategy and also include where to purchase the etfs"
result = trading_forest.run(task)
