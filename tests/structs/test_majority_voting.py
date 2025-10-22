from swarms.structs.agent import Agent
from swarms.structs.majority_voting import MajorityVoting


def test_majority_voting_basic_execution():
    """Test basic MajorityVoting execution with multiple agents"""
    # Create specialized agents with different perspectives
    geographer = Agent(
        agent_name="Geography-Expert",
        agent_description="Expert in geography and world capitals",
        model_name="gpt-4o",
        max_loops=1,
    )

    historian = Agent(
        agent_name="History-Scholar",
        agent_description="Historical and cultural context specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    political_analyst = Agent(
        agent_name="Political-Analyst",
        agent_description="Political and administrative specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Create majority voting system
    mv = MajorityVoting(
        name="Geography-Consensus-System",
        description="Majority voting system for geographical questions",
        agents=[geographer, historian, political_analyst],
        max_loops=1,
        verbose=True,
    )

    # Test execution
    result = mv.run("What is the capital city of France?")
    assert result is not None


def test_majority_voting_multiple_loops():
    """Test MajorityVoting with multiple loops for consensus refinement"""
    # Create agents with different knowledge bases
    trivia_expert = Agent(
        agent_name="Trivia-Expert",
        agent_description="General knowledge and trivia specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    research_analyst = Agent(
        agent_name="Research-Analyst",
        agent_description="Research and fact-checking specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    subject_matter_expert = Agent(
        agent_name="Subject-Matter-Expert",
        agent_description="Deep subject matter expertise specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Create majority voting with multiple loops for iterative refinement
    mv = MajorityVoting(
        name="Multi-Loop-Consensus-System",
        description="Majority voting with iterative consensus refinement",
        agents=[
            trivia_expert,
            research_analyst,
            subject_matter_expert,
        ],
        max_loops=3,  # Allow multiple iterations
        verbose=True,
    )

    # Test multi-loop execution
    result = mv.run(
        "What are the main causes of climate change and what can be done to mitigate them?"
    )
    assert result is not None


def test_majority_voting_business_scenario():
    """Test MajorityVoting in a realistic business scenario"""
    # Create agents representing different business perspectives
    market_strategist = Agent(
        agent_name="Market-Strategist",
        agent_description="Market strategy and competitive analysis specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    financial_analyst = Agent(
        agent_name="Financial-Analyst",
        agent_description="Financial modeling and ROI analysis specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    technical_architect = Agent(
        agent_name="Technical-Architect",
        agent_description="Technical feasibility and implementation specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    risk_manager = Agent(
        agent_name="Risk-Manager",
        agent_description="Risk assessment and compliance specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    operations_expert = Agent(
        agent_name="Operations-Expert",
        agent_description="Operations and implementation specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Create majority voting for business decisions
    mv = MajorityVoting(
        name="Business-Decision-Consensus",
        description="Majority voting system for business strategic decisions",
        agents=[
            market_strategist,
            financial_analyst,
            technical_architect,
            risk_manager,
            operations_expert,
        ],
        max_loops=2,
        verbose=True,
    )

    # Test with complex business decision
    result = mv.run(
        "Should our company invest in developing an AI-powered customer service platform? "
        "Consider market demand, financial implications, technical feasibility, risk factors, "
        "and operational requirements."
    )

    assert result is not None


def test_majority_voting_error_handling():
    """Test MajorityVoting error handling and validation"""
    # Test with empty agents list
    try:
        mv = MajorityVoting(agents=[])
        assert (
            False
        ), "Should have raised ValueError for empty agents list"
    except ValueError as e:
        assert "agents" in str(e).lower() or "empty" in str(e).lower()

    # Test with invalid max_loops
    analyst = Agent(
        agent_name="Test-Analyst",
        agent_description="Test analyst",
        model_name="gpt-4o",
        max_loops=1,
    )

    try:
        mv = MajorityVoting(agents=[analyst], max_loops=0)
        assert (
            False
        ), "Should have raised ValueError for invalid max_loops"
    except ValueError as e:
        assert "max_loops" in str(e).lower() or "0" in str(e)


def test_majority_voting_different_output_types():
    """Test MajorityVoting with different output types"""
    # Create agents for technical analysis
    security_expert = Agent(
        agent_name="Security-Expert",
        agent_description="Cybersecurity and data protection specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    compliance_officer = Agent(
        agent_name="Compliance-Officer",
        agent_description="Regulatory compliance and legal specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    privacy_advocate = Agent(
        agent_name="Privacy-Advocate",
        agent_description="Privacy protection and data rights specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Test different output types
    for output_type in ["dict", "string", "list"]:
        mv = MajorityVoting(
            name=f"Output-Type-Test-{output_type}",
            description=f"Testing output type: {output_type}",
            agents=[
                security_expert,
                compliance_officer,
                privacy_advocate,
            ],
            max_loops=1,
            output_type=output_type,
        )

        result = mv.run(
            "What are the key considerations for implementing GDPR compliance in our data processing systems?"
        )
        assert result is not None
