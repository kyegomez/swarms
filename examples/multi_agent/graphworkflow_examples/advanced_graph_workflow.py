from swarms import Agent
from swarms.structs.graph_workflow import GraphWorkflow


def create_complex_investment_analysis_workflow():
    """
    Creates a sophisticated investment analysis workflow with multiple specialized agents
    working in parallel and series to provide comprehensive market analysis.

    Workflow Structure:
    1. Data Gathering Agent (Entry Point)
    2. Three Parallel Research Agents:
       - Fundamental Analysis Agent
       - Technical Analysis Agent
       - Sentiment Analysis Agent
    3. Risk Assessment Agent (runs in parallel with research agents)
    4. Market Context Agent (analyzes broader market conditions)
    5. Synthesis Agent (combines all research outputs)
    6. Final Recommendation Agent (End Point)

    Returns:
        GraphWorkflow: Configured workflow ready for execution
    """

    # Create specialized agents with detailed system prompts
    data_gathering_agent = Agent(
        agent_name="DataGatheringAgent",
        model_name="gpt-4.1",
        max_loops=1,
        system_prompt="""You are a financial data gathering specialist. Your role is to:
        1. Identify and collect relevant financial data for the given investment target
        2. Gather recent news, earnings reports, and market data
        3. Compile key financial metrics and ratios
        4. Provide a comprehensive data foundation for other analysts
        5. Structure your output clearly for downstream analysis
        
        Focus on accuracy, recency, and relevance of data. Always cite sources when possible.""",
        verbose=True,
    )

    fundamental_analysis_agent = Agent(
        agent_name="FundamentalAnalysisAgent",
        model_name="gpt-4.1",
        max_loops=1,
        system_prompt="""You are a fundamental analysis expert. Your role is to:
        1. Analyze company financials, business model, and competitive position
        2. Evaluate management quality and corporate governance
        3. Assess industry trends and market position
        4. Calculate intrinsic value using various valuation methods
        5. Identify fundamental strengths and weaknesses
        
        Base your analysis on solid financial principles and provide quantitative backing for your conclusions.""",
        verbose=True,
    )

    technical_analysis_agent = Agent(
        agent_name="TechnicalAnalysisAgent",
        model_name="gpt-4.1",
        max_loops=1,
        system_prompt="""You are a technical analysis specialist. Your role is to:
        1. Analyze price charts, trends, and trading patterns
        2. Identify support and resistance levels
        3. Evaluate momentum indicators and trading signals
        4. Assess volume patterns and market sentiment
        5. Provide entry/exit timing recommendations
        
        Use established technical analysis principles and explain your reasoning clearly.""",
        verbose=True,
    )

    sentiment_analysis_agent = Agent(
        agent_name="SentimentAnalysisAgent",
        model_name="gpt-4.1",
        max_loops=1,
        system_prompt="""You are a market sentiment analysis expert. Your role is to:
        1. Analyze social media sentiment and retail investor behavior
        2. Evaluate institutional investor positioning and flows
        3. Assess news sentiment and media coverage
        4. Monitor options flow and derivatives positioning
        5. Gauge overall market psychology and positioning
        
        Provide insights into market sentiment trends and their potential impact.""",
        verbose=True,
    )

    risk_assessment_agent = Agent(
        agent_name="RiskAssessmentAgent",
        model_name="gpt-4.1",
        max_loops=1,
        system_prompt="""You are a risk management specialist. Your role is to:
        1. Identify and quantify various risk factors (market, credit, liquidity, operational)
        2. Analyze historical volatility and correlation patterns
        3. Assess downside scenarios and tail risks
        4. Evaluate portfolio impact and position sizing considerations
        5. Recommend risk mitigation strategies
        
        Provide comprehensive risk analysis with quantitative metrics where possible.""",
        verbose=True,
    )

    market_context_agent = Agent(
        agent_name="MarketContextAgent",
        model_name="gpt-4.1",
        max_loops=1,
        system_prompt="""You are a macro market analysis expert. Your role is to:
        1. Analyze broader market conditions and economic environment
        2. Evaluate sector rotation and style preferences
        3. Assess correlation with market indices and sector peers
        4. Consider geopolitical and regulatory factors
        5. Provide market timing and allocation context
        
        Focus on how broader market conditions might impact the specific investment.""",
        verbose=True,
    )

    synthesis_agent = Agent(
        agent_name="SynthesisAgent",
        model_name="gpt-4.1",
        max_loops=1,
        system_prompt="""You are an investment analysis synthesizer. Your role is to:
        1. Integrate findings from fundamental, technical, and sentiment analysis
        2. Reconcile conflicting viewpoints and identify consensus areas
        3. Weight different analysis components based on current market conditions
        4. Identify the most compelling investment thesis
        5. Highlight key risks and opportunities
        
        Provide a balanced synthesis that considers all analytical perspectives.""",
        verbose=True,
    )

    recommendation_agent = Agent(
        agent_name="FinalRecommendationAgent",
        model_name="gpt-4.1",
        max_loops=1,
        system_prompt="""You are the final investment decision maker. Your role is to:
        1. Review all analysis and synthesis from the team
        2. Make a clear investment recommendation (BUY/HOLD/SELL)
        3. Provide specific entry/exit criteria and price targets
        4. Recommend position sizing and risk management approach
        5. Outline monitoring criteria and review timeline
        
        Provide actionable investment guidance with clear rationale and risk considerations.""",
        verbose=True,
    )

    # Create the workflow
    workflow = GraphWorkflow(
        name="ComplexInvestmentAnalysisWorkflow",
        description="A comprehensive multi-agent investment analysis system with parallel processing and sophisticated agent collaboration",
        verbose=True,
        auto_compile=True,
    )

    # Add all agents as nodes
    agents = [
        data_gathering_agent,
        fundamental_analysis_agent,
        technical_analysis_agent,
        sentiment_analysis_agent,
        risk_assessment_agent,
        market_context_agent,
        synthesis_agent,
        recommendation_agent,
    ]

    for agent in agents:
        workflow.add_node(agent)

    # Define complex edge relationships
    # Stage 1: Data gathering feeds into all analysis agents
    workflow.add_edge(
        "DataGatheringAgent", "FundamentalAnalysisAgent"
    )
    workflow.add_edge("DataGatheringAgent", "TechnicalAnalysisAgent")
    workflow.add_edge("DataGatheringAgent", "SentimentAnalysisAgent")
    workflow.add_edge("DataGatheringAgent", "RiskAssessmentAgent")
    workflow.add_edge("DataGatheringAgent", "MarketContextAgent")

    # Stage 2: All analysis agents feed into synthesis
    workflow.add_edge("FundamentalAnalysisAgent", "SynthesisAgent")
    workflow.add_edge("TechnicalAnalysisAgent", "SynthesisAgent")
    workflow.add_edge("SentimentAnalysisAgent", "SynthesisAgent")

    # Stage 3: Synthesis and risk/context feed into final recommendation
    workflow.add_edge("SynthesisAgent", "FinalRecommendationAgent")
    workflow.add_edge(
        "RiskAssessmentAgent", "FinalRecommendationAgent"
    )
    workflow.add_edge(
        "MarketContextAgent", "FinalRecommendationAgent"
    )

    # Set explicit entry and end points
    workflow.set_entry_points(["DataGatheringAgent"])
    workflow.set_end_points(["FinalRecommendationAgent"])

    return workflow


# def create_parallel_research_workflow():
#     """
#     Creates a parallel research workflow demonstrating multiple entry points
#     and complex convergence patterns.

#     Returns:
#         GraphWorkflow: Configured parallel research workflow
#     """

#     # Create research agents for different domains
#     academic_researcher = Agent(
#         agent_name="AcademicResearcher",
#         model_name="gpt-4.1",
#         max_loops=1,
#         system_prompt="You are an academic researcher specializing in peer-reviewed literature analysis. Focus on scientific papers, studies, and academic sources.",
#         verbose=True,
#     )

#     industry_analyst = Agent(
#         agent_name="IndustryAnalyst",
#         model_name="gpt-4.1",
#         max_loops=1,
#         system_prompt="You are an industry analyst focusing on market reports, industry trends, and commercial applications.",
#         verbose=True,
#     )

#     news_researcher = Agent(
#         agent_name="NewsResearcher",
#         model_name="gpt-4.1",
#         max_loops=1,
#         system_prompt="You are a news researcher specializing in current events, breaking news, and recent developments.",
#         verbose=True,
#     )

#     data_scientist = Agent(
#         agent_name="DataScientist",
#         model_name="gpt-4.1",
#         max_loops=1,
#         system_prompt="You are a data scientist focusing on quantitative analysis, statistical patterns, and data-driven insights.",
#         verbose=True,
#     )

#     synthesizer = Agent(
#         agent_name="ResearchSynthesizer",
#         model_name="gpt-4.1",
#         max_loops=1,
#         system_prompt="You are a research synthesizer who combines insights from multiple research domains into coherent conclusions.",
#         verbose=True,
#     )

#     quality_checker = Agent(
#         agent_name="QualityChecker",
#         model_name="gpt-4.1",
#         max_loops=1,
#         system_prompt="You are a quality assurance specialist who validates research findings and identifies potential gaps or biases.",
#         verbose=True,
#     )

#     # Create workflow with multiple entry points
#     workflow = GraphWorkflow(
#         name="ParallelResearchWorkflow",
#         description="A parallel research workflow with multiple independent entry points converging to synthesis",
#         verbose=True,
#     )

#     # Add all agents
#     for agent in [
#         academic_researcher,
#         industry_analyst,
#         news_researcher,
#         data_scientist,
#         synthesizer,
#         quality_checker,
#     ]:
#         workflow.add_node(agent)

#     # Create convergence pattern - all researchers feed into synthesizer
#     workflow.add_edge("AcademicResearcher", "ResearchSynthesizer")
#     workflow.add_edge("IndustryAnalyst", "ResearchSynthesizer")
#     workflow.add_edge("NewsResearcher", "ResearchSynthesizer")
#     workflow.add_edge("DataScientist", "ResearchSynthesizer")

#     # Synthesizer feeds into quality checker
#     workflow.add_edge("ResearchSynthesizer", "QualityChecker")

#     # Set multiple entry points (parallel execution)
#     workflow.set_entry_points(
#         [
#             "AcademicResearcher",
#             "IndustryAnalyst",
#             "NewsResearcher",
#             "DataScientist",
#         ]
#     )
#     workflow.set_end_points(["QualityChecker"])

#     return workflow


# def demonstrate_complex_workflows():
#     """
#     Demonstrates both complex workflow examples with different tasks.
#     """
#     investment_workflow = (
#         create_complex_investment_analysis_workflow()
#     )

#     # Visualize the workflow structure
#     investment_workflow.visualize()

#     # Run the investment analysis
#     investment_task = """
#     Analyze Tesla (TSLA) stock as a potential investment opportunity.
#     Consider the company's fundamentals, technical chart patterns, market sentiment,
#     risk factors, and broader market context. Provide a comprehensive investment
#     recommendation with specific entry/exit criteria.
#     """

#     investment_results = investment_workflow.run(task=investment_task)

#     for agent_name, result in investment_results.items():
#         print(f"\n🤖 {agent_name}:")
#         print(f"{result[:300]}{'...' if len(result) > 300 else ''}")

#     research_workflow = create_parallel_research_workflow()

#     # Run the research analysis
#     research_task = """
#     Research the current state and future prospects of quantum computing.
#     Examine academic progress, industry developments, recent news, and
#     quantitative trends. Provide a comprehensive analysis of the field's
#     current status and trajectory.
#     """

#     research_results = research_workflow.run(task=research_task)

#     for agent_name, result in research_results.items():
#         print(f"\n🤖 {agent_name}:")
#         print(f"{result[:300]}{'...' if len(result) > 300 else ''}")


# if __name__ == "__main__":
#     # Run the comprehensive demonstration
#     demonstrate_complex_workflows()


if __name__ == "__main__":
    workflow = create_complex_investment_analysis_workflow()
    workflow.visualize()
    # workflow.run(
    #     task="Analyze Tesla (TSLA) stock as a potential investment opportunity. Consider the company's fundamentals, technical chart patterns, market sentiment, risk factors, and broader market context. Provide a comprehensive investment recommendation with specific entry/exit criteria."
    # )
