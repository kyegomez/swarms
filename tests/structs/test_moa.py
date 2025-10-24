from swarms.structs.mixture_of_agents import MixtureOfAgents
from swarms.structs.agent import Agent


def test_mixture_of_agents_basic_initialization():
    """Test basic MixtureOfAgents initialization with multiple agents"""
    # Create multiple specialized agents
    research_agent = Agent(
        agent_name="Research-Specialist",
        agent_description="Specialist in research and data collection",
        model_name="gpt-4o",
        max_loops=1,
    )

    analysis_agent = Agent(
        agent_name="Analysis-Expert",
        agent_description="Expert in data analysis and insights",
        model_name="gpt-4o",
        max_loops=1,
    )

    strategy_agent = Agent(
        agent_name="Strategy-Consultant",
        agent_description="Strategy and planning consultant",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Create aggregator agent
    aggregator = Agent(
        agent_name="Aggregator-Agent",
        agent_description="Agent that aggregates responses from other agents",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Create mixture of agents
    moa = MixtureOfAgents(
        name="Business-Analysis-Mixture",
        description="Mixture of agents for comprehensive business analysis",
        agents=[research_agent, analysis_agent, strategy_agent],
        aggregator_agent=aggregator,
        layers=3,
        max_loops=1,
    )

    # Verify initialization
    assert moa.name == "Business-Analysis-Mixture"
    assert (
        moa.description
        == "Mixture of agents for comprehensive business analysis"
    )
    assert len(moa.agents) == 3
    assert moa.aggregator_agent == aggregator
    assert moa.layers == 3
    assert moa.max_loops == 1


def test_mixture_of_agents_execution():
    """Test MixtureOfAgents execution with multiple agents"""
    # Create diverse agents for different perspectives
    market_analyst = Agent(
        agent_name="Market-Analyst",
        agent_description="Market analysis and trend specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    technical_expert = Agent(
        agent_name="Technical-Expert",
        agent_description="Technical feasibility and implementation specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    financial_analyst = Agent(
        agent_name="Financial-Analyst",
        agent_description="Financial modeling and ROI specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    risk_assessor = Agent(
        agent_name="Risk-Assessor",
        agent_description="Risk assessment and mitigation specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Create aggregator for synthesis
    aggregator = Agent(
        agent_name="Executive-Summary-Agent",
        agent_description="Executive summary and recommendation specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Create mixture of agents
    moa = MixtureOfAgents(
        name="Comprehensive-Evaluation-Mixture",
        description="Mixture of agents for comprehensive business evaluation",
        agents=[
            market_analyst,
            technical_expert,
            financial_analyst,
            risk_assessor,
        ],
        aggregator_agent=aggregator,
        layers=2,
        max_loops=1,
    )

    # Test execution
    result = moa.run(
        "Evaluate the feasibility of launching an AI-powered healthcare platform"
    )
    assert result is not None


def test_mixture_of_agents_multiple_layers():
    """Test MixtureOfAgents with multiple layers"""
    # Create agents for layered analysis
    data_collector = Agent(
        agent_name="Data-Collector",
        agent_description="Data collection and research specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    pattern_analyzer = Agent(
        agent_name="Pattern-Analyzer",
        agent_description="Pattern recognition and analysis specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    insight_generator = Agent(
        agent_name="Insight-Generator",
        agent_description="Insight generation and interpretation specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Create aggregator
    final_aggregator = Agent(
        agent_name="Final-Aggregator",
        agent_description="Final aggregation and conclusion specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Create mixture with multiple layers for deeper analysis
    moa = MixtureOfAgents(
        name="Multi-Layer-Analysis-Mixture",
        description="Mixture of agents with multiple analysis layers",
        agents=[data_collector, pattern_analyzer, insight_generator],
        aggregator_agent=final_aggregator,
        layers=4,
        max_loops=1,
    )

    # Test multi-layer execution
    result = moa.run(
        "Analyze customer behavior patterns and provide strategic insights"
    )
    assert result is not None


def test_mixture_of_agents_error_handling():
    """Test MixtureOfAgents error handling and validation"""
    # Test with empty agents list
    try:
        moa = MixtureOfAgents(agents=[])
        assert (
            False
        ), "Should have raised ValueError for empty agents list"
    except ValueError as e:
        assert "No agents provided" in str(e)

    # Test with invalid aggregator system prompt
    analyst = Agent(
        agent_name="Test-Analyst",
        agent_description="Test analyst",
        model_name="gpt-4o",
        max_loops=1,
    )

    try:
        moa = MixtureOfAgents(
            agents=[analyst], aggregator_system_prompt=""
        )
        assert (
            False
        ), "Should have raised ValueError for empty system prompt"
    except ValueError as e:
        assert "No aggregator system prompt" in str(e)


def test_mixture_of_agents_real_world_scenario():
    """Test MixtureOfAgents in a realistic business scenario"""
    # Create agents representing different business functions
    marketing_director = Agent(
        agent_name="Marketing-Director",
        agent_description="Senior marketing director with market expertise",
        model_name="gpt-4o",
        max_loops=1,
    )

    product_manager = Agent(
        agent_name="Product-Manager",
        agent_description="Product strategy and development manager",
        model_name="gpt-4o",
        max_loops=1,
    )

    engineering_lead = Agent(
        agent_name="Engineering-Lead",
        agent_description="Senior engineering and technical architecture lead",
        model_name="gpt-4o",
        max_loops=1,
    )

    sales_executive = Agent(
        agent_name="Sales-Executive",
        agent_description="Enterprise sales and customer relationship executive",
        model_name="gpt-4o",
        max_loops=1,
    )

    legal_counsel = Agent(
        agent_name="Legal-Counsel",
        agent_description="Legal compliance and regulatory counsel",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Create aggregator for executive decision making
    executive_aggregator = Agent(
        agent_name="Executive-Decision-Maker",
        agent_description="Executive decision maker and strategic aggregator",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Create comprehensive mixture of agents
    moa = MixtureOfAgents(
        name="Executive-Board-Mixture",
        description="Mixture of agents representing executive board for strategic decisions",
        agents=[
            marketing_director,
            product_manager,
            engineering_lead,
            sales_executive,
            legal_counsel,
        ],
        aggregator_agent=executive_aggregator,
        layers=3,
        max_loops=1,
    )

    # Test with complex business scenario
    result = moa.run(
        "Develop a comprehensive go-to-market strategy for our new AI-powered enterprise platform. "
        "Consider market positioning, technical requirements, competitive landscape, sales channels, "
        "and legal compliance requirements."
    )

    assert result is not None
