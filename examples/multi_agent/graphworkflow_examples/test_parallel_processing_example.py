"""
Comprehensive example demonstrating GraphWorkflow parallel processing capabilities.
This showcases fan-out, fan-in, and parallel chain patterns for maximum efficiency.
"""

from swarms import Agent
from swarms.structs.graph_workflow import GraphWorkflow


def create_advanced_financial_analysis_workflow():
    """
    Create a sophisticated financial analysis workflow demonstrating
    all parallel processing patterns for maximum efficiency.

    Workflow Architecture:
    1. Data Collection (Entry Point)
    2. Fan-out to 3 Parallel Data Processors
    3. Fan-out to 4 Parallel Analysis Specialists
    4. Fan-in to Synthesis Agent
    5. Final Recommendation (End Point)
    """

    # === Data Collection Layer ===
    data_collector = Agent(
        agent_name="DataCollector",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="""You are a financial data collection specialist. Your role is to:
        1. Gather comprehensive market data for the target investment
        2. Collect recent news, earnings reports, and analyst ratings
        3. Compile key financial metrics and historical performance data
        4. Structure the data clearly for downstream parallel analysis
        
        Provide comprehensive data that multiple specialists can analyze simultaneously.""",
        verbose=False,
    )

    # === Parallel Data Processing Layer ===
    market_data_processor = Agent(
        agent_name="MarketDataProcessor",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="""You are a market data processing specialist. Focus on:
        1. Market price movements and trading volumes
        2. Technical indicators and chart patterns
        3. Market sentiment and momentum signals
        4. Sector and peer comparison data
        
        Process raw market data into analysis-ready insights.""",
        verbose=False,
    )

    fundamental_data_processor = Agent(
        agent_name="FundamentalDataProcessor",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="""You are a fundamental data processing specialist. Focus on:
        1. Financial statements and accounting metrics
        2. Business model and competitive positioning
        3. Management quality and corporate governance
        4. Industry trends and regulatory environment
        
        Process fundamental data into comprehensive business analysis.""",
        verbose=False,
    )

    news_data_processor = Agent(
        agent_name="NewsDataProcessor",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="""You are a news and sentiment data processor. Focus on:
        1. Recent news events and their market impact
        2. Analyst opinions and rating changes
        3. Social media sentiment and retail investor behavior
        4. Institutional investor positioning and flows
        
        Process news and sentiment data into actionable insights.""",
        verbose=False,
    )

    # === Parallel Analysis Specialists Layer ===
    technical_analyst = Agent(
        agent_name="TechnicalAnalyst",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="""You are a technical analysis expert specializing in:
        1. Chart pattern analysis and trend identification
        2. Support and resistance level analysis
        3. Momentum and oscillator interpretation
        4. Entry and exit timing recommendations
        
        Provide detailed technical analysis with specific price targets.""",
        verbose=False,
    )

    fundamental_analyst = Agent(
        agent_name="FundamentalAnalyst",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="""You are a fundamental analysis expert specializing in:
        1. Intrinsic value calculation using multiple methods
        2. Financial ratio analysis and peer comparison
        3. Business model evaluation and competitive moats
        4. Growth prospects and risk assessment
        
        Provide comprehensive fundamental analysis with valuation estimates.""",
        verbose=False,
    )

    risk_analyst = Agent(
        agent_name="RiskAnalyst",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="""You are a risk management specialist focusing on:
        1. Quantitative risk metrics (VaR, volatility, correlations)
        2. Scenario analysis and stress testing
        3. Downside protection and tail risk assessment
        4. Portfolio impact and position sizing recommendations
        
        Provide comprehensive risk analysis with mitigation strategies.""",
        verbose=False,
    )

    esg_analyst = Agent(
        agent_name="ESGAnalyst",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="""You are an ESG (Environmental, Social, Governance) specialist focusing on:
        1. Environmental impact and sustainability practices
        2. Social responsibility and stakeholder relations
        3. Corporate governance and ethical leadership
        4. Regulatory compliance and reputational risks
        
        Provide comprehensive ESG analysis and scoring.""",
        verbose=False,
    )

    # === Synthesis and Final Decision Layer ===
    synthesis_agent = Agent(
        agent_name="SynthesisAgent",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="""You are an investment synthesis specialist. Your role is to:
        1. Integrate all analysis from technical, fundamental, risk, and ESG specialists
        2. Reconcile conflicting viewpoints and identify consensus areas
        3. Weight different analysis components based on market conditions
        4. Identify the most compelling investment thesis and key risks
        
        Provide a balanced synthesis that considers all analytical perspectives.""",
        verbose=False,
    )

    portfolio_manager = Agent(
        agent_name="PortfolioManager",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="""You are the final investment decision maker. Your role is to:
        1. Review all synthesis and specialist analysis
        2. Make clear investment recommendations (BUY/HOLD/SELL)
        3. Provide specific entry/exit criteria and price targets
        4. Recommend position sizing and risk management approach
        5. Outline monitoring criteria and review timeline
        
        Provide actionable investment guidance with clear rationale.""",
        verbose=False,
    )

    # === Create Workflow ===
    workflow = GraphWorkflow(
        name="Advanced-Parallel-Financial-Analysis",
        description="Sophisticated multi-agent financial analysis workflow demonstrating fan-out, fan-in, and parallel processing patterns for maximum efficiency and comprehensive analysis coverage.",
        max_loops=1,
        verbose=True,
        auto_compile=True,
        task="Analyze Apple Inc. (AAPL) as a potential investment opportunity with comprehensive parallel analysis covering technical, fundamental, risk, and ESG factors.",
    )

    # Add all agents
    agents = [
        data_collector,
        market_data_processor,
        fundamental_data_processor,
        news_data_processor,
        technical_analyst,
        fundamental_analyst,
        risk_analyst,
        esg_analyst,
        synthesis_agent,
        portfolio_manager,
    ]

    for agent in agents:
        workflow.add_node(agent)

    # === Create Parallel Processing Architecture ===

    # Stage 1: Data Collection feeds into parallel processors (Fan-out)
    workflow.add_edges_from_source(
        "DataCollector",
        [
            "MarketDataProcessor",
            "FundamentalDataProcessor",
            "NewsDataProcessor",
        ],
    )

    # Stage 2: Each processor feeds specific analysts (Targeted Fan-out)
    workflow.add_edge("MarketDataProcessor", "TechnicalAnalyst")
    workflow.add_edge(
        "FundamentalDataProcessor", "FundamentalAnalyst"
    )
    workflow.add_edge("NewsDataProcessor", "ESGAnalyst")

    # Stage 3: All processors also feed risk analyst (Additional Fan-in)
    workflow.add_edges_to_target(
        [
            "MarketDataProcessor",
            "FundamentalDataProcessor",
            "NewsDataProcessor",
        ],
        "RiskAnalyst",
    )

    # Stage 4: All specialists feed synthesis (Major Fan-in)
    workflow.add_edges_to_target(
        [
            "TechnicalAnalyst",
            "FundamentalAnalyst",
            "RiskAnalyst",
            "ESGAnalyst",
        ],
        "SynthesisAgent",
    )

    # Stage 5: Synthesis feeds portfolio manager (Final Decision)
    workflow.add_edge("SynthesisAgent", "PortfolioManager")

    return workflow


# def create_parallel_research_workflow():
#     """
#     Create a parallel research workflow using the new from_spec syntax
#     that supports parallel patterns.
#     """

#     # Create research agents
#     web_researcher = Agent(
#         agent_name="WebResearcher",
#         model_name="gpt-4o-mini",
#         max_loops=1,
#         system_prompt="You are a web research specialist. Focus on online sources, news, and current information.",
#         verbose=False,
#     )

#     academic_researcher = Agent(
#         agent_name="AcademicResearcher",
#         model_name="gpt-4o-mini",
#         max_loops=1,
#         system_prompt="You are an academic research specialist. Focus on peer-reviewed papers and scholarly sources.",
#         verbose=False,
#     )

#     market_researcher = Agent(
#         agent_name="MarketResearcher",
#         model_name="gpt-4o-mini",
#         max_loops=1,
#         system_prompt="You are a market research specialist. Focus on industry reports and market analysis.",
#         verbose=False,
#     )

#     analyst1 = Agent(
#         agent_name="Analyst1",
#         model_name="gpt-4o-mini",
#         max_loops=1,
#         system_prompt="You are Analysis Specialist 1. Provide quantitative analysis.",
#         verbose=False,
#     )

#     analyst2 = Agent(
#         agent_name="Analyst2",
#         model_name="gpt-4o-mini",
#         max_loops=1,
#         system_prompt="You are Analysis Specialist 2. Provide qualitative analysis.",
#         verbose=False,
#     )

#     synthesizer = Agent(
#         agent_name="ResearchSynthesizer",
#         model_name="gpt-4o-mini",
#         max_loops=1,
#         system_prompt="You are a research synthesizer. Combine all research into comprehensive conclusions.",
#         verbose=False,
#     )

#     # Use from_spec with new parallel edge syntax
#     workflow = GraphWorkflow.from_spec(
#         agents=[web_researcher, academic_researcher, market_researcher, analyst1, analyst2, synthesizer],
#         edges=[
#             # Fan-out: Each researcher feeds both analysts (parallel chain)
#             (["WebResearcher", "AcademicResearcher", "MarketResearcher"], ["Analyst1", "Analyst2"]),
#             # Fan-in: Both analysts feed synthesizer
#             (["Analyst1", "Analyst2"], "ResearchSynthesizer")
#         ],
#         name="Parallel-Research-Workflow",
#         description="Parallel research workflow using advanced edge syntax",
#         max_loops=1,
#         verbose=True,
#         task="Research the future of renewable energy technology and market opportunities"
#     )

#     return workflow


# def demonstrate_parallel_patterns():
#     """
#     Demonstrate all parallel processing patterns and their benefits.
#     """
#     print("🚀 ADVANCED PARALLEL PROCESSING DEMONSTRATION")
#     print("=" * 70)

#     # === Advanced Financial Analysis ===
#     print("\n💰 ADVANCED FINANCIAL ANALYSIS WORKFLOW")
#     print("-" * 50)

#     financial_workflow = create_advanced_financial_analysis_workflow()

#     print("\n📊 Creating Graphviz Visualization...")
#     try:
#         # Create PNG visualization
#         png_output = financial_workflow.visualize(
#             output_path="financial_workflow_graph",
#             format="png",
#             view=False,  # Don't auto-open for demo
#             show_parallel_patterns=True
#         )
#         print(f"✅ Financial workflow visualization saved: {png_output}")

#         # Create SVG for web use
#         svg_output = financial_workflow.visualize(
#             output_path="financial_workflow_web",
#             format="svg",
#             view=False,
#             show_parallel_patterns=True
#         )
#         print(f"✅ Web-ready SVG visualization saved: {svg_output}")

#     except Exception as e:
#         print(f"⚠️ Graphviz visualization failed, using fallback: {e}")
#         financial_workflow.visualize()

#     print(f"\n📈 Workflow Architecture:")
#     print(f"  Total Agents: {len(financial_workflow.nodes)}")
#     print(f"  Total Connections: {len(financial_workflow.edges)}")
#     print(f"  Parallel Layers: {len(financial_workflow._sorted_layers) if financial_workflow._compiled else 'Not compiled'}")

#     # Show compilation benefits
#     status = financial_workflow.get_compilation_status()
#     print(f"  Compilation Status: {status['is_compiled']}")
#     print(f"  Cache Efficient: {status['cache_efficient']}")

#     # === Parallel Research Workflow ===
#     print("\n\n📚 PARALLEL RESEARCH WORKFLOW (from_spec)")
#     print("-" * 50)

#     research_workflow = create_parallel_research_workflow()

#     print("\n📊 Creating Research Workflow Visualization...")
#     try:
#         # Create circular layout for research workflow
#         research_output = research_workflow.visualize(
#             output_path="research_workflow_graph",
#             format="png",
#             view=False,
#             engine="circo",  # Circular layout for smaller graphs
#             show_parallel_patterns=True
#         )
#         print(f"✅ Research workflow visualization saved: {research_output}")
#     except Exception as e:
#         print(f"⚠️ Graphviz visualization failed, using fallback: {e}")
#         research_workflow.visualize()

#     print(f"\n📈 Research Workflow Architecture:")
#     print(f"  Total Agents: {len(research_workflow.nodes)}")
#     print(f"  Total Connections: {len(research_workflow.edges)}")
#     print(f"  Entry Points: {research_workflow.entry_points}")
#     print(f"  End Points: {research_workflow.end_points}")

#     # === Performance Analysis ===
#     print("\n\n⚡ PARALLEL PROCESSING BENEFITS")
#     print("-" * 50)

#     print("🔀 Pattern Analysis:")

#     # Analyze financial workflow patterns
#     fin_fan_out = {}
#     fin_fan_in = {}

#     for edge in financial_workflow.edges:
#         # Track fan-out
#         if edge.source not in fin_fan_out:
#             fin_fan_out[edge.source] = []
#         fin_fan_out[edge.source].append(edge.target)

#         # Track fan-in
#         if edge.target not in fin_fan_in:
#             fin_fan_in[edge.target] = []
#         fin_fan_in[edge.target].append(edge.source)

#     fan_out_count = sum(1 for targets in fin_fan_out.values() if len(targets) > 1)
#     fan_in_count = sum(1 for sources in fin_fan_in.values() if len(sources) > 1)
#     parallel_nodes = sum(len(targets) for targets in fin_fan_out.values() if len(targets) > 1)

#     print(f"  Financial Workflow:")
#     print(f"    🔀 Fan-out Patterns: {fan_out_count}")
#     print(f"    🔀 Fan-in Patterns: {fan_in_count}")
#     print(f"    ⚡ Parallel Execution Nodes: {parallel_nodes}")
#     print(f"    🎯 Efficiency Gain: ~{(parallel_nodes / len(financial_workflow.nodes)) * 100:.1f}% parallel processing")

#     # === Export Examples ===
#     print("\n\n💾 WORKFLOW EXPORT EXAMPLE")
#     print("-" * 50)

#     try:
#         # Save financial workflow
#         saved_path = financial_workflow.save_to_file(
#             "advanced_financial_workflow.json",
#             include_runtime_state=True,
#             overwrite=True
#         )
#         print(f"✅ Financial workflow saved to: {saved_path}")

#         # Export summary
#         summary = financial_workflow.export_summary()
#         print(f"\n📋 Workflow Summary:")
#         print(f"  Agents: {len(summary['agents'])}")
#         print(f"  Connections: {len(summary['connections'])}")
#         print(f"  Parallel Patterns Detected: {fan_out_count + fan_in_count}")

#     except Exception as e:
#         print(f"⚠️ Export failed: {e}")

#     print("\n\n🎯 PARALLEL PROCESSING SUMMARY")
#     print("=" * 70)
#     print("✅ Fan-out patterns: One agent output distributed to multiple agents")
#     print("✅ Fan-in patterns: Multiple agent outputs converged to one agent")
#     print("✅ Parallel chains: Multiple sources connected to multiple targets")
#     print("✅ Enhanced visualization: Shows parallel patterns clearly")
#     print("✅ Compilation caching: Optimized execution for complex graphs")
#     print("✅ Flexible from_spec syntax: Easy parallel workflow creation")
#     print("✅ Maximum efficiency: Parallel processing instead of sequential chains")


# if __name__ == "__main__":
#     demonstrate_parallel_patterns()

if __name__ == "__main__":
    workflow = create_advanced_financial_analysis_workflow()
    workflow.visualize(
        output_path="advanced_financial_analysis_workflow",
        format="png",
        view=True,
        show_parallel_patterns=True,
    )
