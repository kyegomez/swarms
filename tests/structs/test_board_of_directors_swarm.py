"""
Comprehensive test suite for Board of Directors Swarm.

This module contains extensive tests for the Board of Directors swarm implementation,
covering all aspects including initialization, board operations, task execution,
error handling, and performance characteristics.

Tests follow the example.py pattern with real agents and multiple agent scenarios.
"""

import pytest
from swarms.structs.board_of_directors_swarm import (
    BoardOfDirectorsSwarm,
)
from swarms.structs.agent import Agent


@pytest.fixture
def sample_agents():
    """Create sample real agents for testing."""
    agents = []
    for i in range(5):
        agent = Agent(
            agent_name=f"Board-Member-{i+1}",
            agent_description=f"Board member {i+1} with expertise in strategic decision making",
            model_name="gpt-4o",
            max_loops=1,
        )
        agents.append(agent)
    return agents


@pytest.fixture
def basic_board_swarm(sample_agents):
    """Create a basic Board of Directors swarm for testing."""
    return BoardOfDirectorsSwarm(
        name="Test-Board-Swarm",
        description="Test board of directors swarm for comprehensive testing",
        agents=sample_agents,
        max_loops=1,
        verbose=True,
    )


def test_board_of_directors_swarm_basic_initialization(
    basic_board_swarm,
):
    """Test basic BoardOfDirectorsSwarm initialization with multiple agents"""
    # Verify initialization
    assert basic_board_swarm.name == "Test-Board-Swarm"
    assert (
        basic_board_swarm.description
        == "Test board of directors swarm for comprehensive testing"
    )
    assert len(basic_board_swarm.agents) == 5
    assert basic_board_swarm.max_loops == 1
    assert basic_board_swarm.verbose is True
    assert basic_board_swarm.board_model_name == "gpt-4o-mini"
    assert basic_board_swarm.decision_threshold == 0.6
    assert basic_board_swarm.enable_voting is True
    assert basic_board_swarm.enable_consensus is True


def test_board_of_directors_swarm_execution(basic_board_swarm):
    """Test BoardOfDirectorsSwarm execution with multiple board members"""
    # Test execution
    result = basic_board_swarm.run(
        "Develop a strategic plan for entering the renewable energy market. "
        "Consider market opportunities, competitive landscape, technical requirements, "
        "and regulatory compliance."
    )

    assert result is not None


def test_board_of_directors_swarm_with_custom_configuration():
    """Test BoardOfDirectorsSwarm with custom configuration"""
    # Create specialized agents for different board roles
    ceo = Agent(
        agent_name="CEO",
        agent_description="Chief Executive Officer with overall strategic vision",
        model_name="gpt-4o",
        max_loops=1,
    )

    cfo = Agent(
        agent_name="CFO",
        agent_description="Chief Financial Officer with financial expertise",
        model_name="gpt-4o",
        max_loops=1,
    )

    cto = Agent(
        agent_name="CTO",
        agent_description="Chief Technology Officer with technical expertise",
        model_name="gpt-4o",
        max_loops=1,
    )

    cmo = Agent(
        agent_name="CMO",
        agent_description="Chief Marketing Officer with market expertise",
        model_name="gpt-4o",
        max_loops=1,
    )

    legal_counsel = Agent(
        agent_name="Legal-Counsel",
        agent_description="Chief Legal Officer with regulatory expertise",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Create board swarm with custom configuration
    board_swarm = BoardOfDirectorsSwarm(
        name="Executive-Board-Swarm",
        description="Executive board for strategic enterprise decisions",
        agents=[ceo, cfo, cto, cmo, legal_counsel],
        max_loops=2,
        decision_threshold=0.7,
        enable_voting=True,
        enable_consensus=True,
        verbose=True,
    )

    # Test execution with complex scenario
    result = board_swarm.run(
        "Evaluate the acquisition of a competitor in the AI space. "
        "Consider financial implications, technical integration challenges, "
        "market positioning, legal considerations, and overall strategic fit."
    )

    assert result is not None


def test_board_of_directors_swarm_error_handling():
    """Test BoardOfDirectorsSwarm error handling and validation"""
    # Test with empty agents list
    try:
        board_swarm = BoardOfDirectorsSwarm(agents=[])
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
        board_swarm = BoardOfDirectorsSwarm(
            agents=[analyst], max_loops=0
        )
        assert (
            False
        ), "Should have raised ValueError for invalid max_loops"
    except ValueError as e:
        assert "max_loops" in str(e).lower() or "0" in str(e)


def test_board_of_directors_swarm_real_world_scenario():
    """Test BoardOfDirectorsSwarm in a realistic business scenario"""
    # Create agents representing different C-suite executives
    chief_strategy_officer = Agent(
        agent_name="Chief-Strategy-Officer",
        agent_description="Chief Strategy Officer with expertise in corporate strategy and market analysis",
        model_name="gpt-4o",
        max_loops=1,
    )

    chief_technology_officer = Agent(
        agent_name="Chief-Technology-Officer",
        agent_description="Chief Technology Officer with deep technical expertise and innovation focus",
        model_name="gpt-4o",
        max_loops=1,
    )

    chief_financial_officer = Agent(
        agent_name="Chief-Financial-Officer",
        agent_description="Chief Financial Officer with expertise in financial planning and risk management",
        model_name="gpt-4o",
        max_loops=1,
    )

    chief_operating_officer = Agent(
        agent_name="Chief-Operating-Officer",
        agent_description="Chief Operating Officer with expertise in operations and implementation",
        model_name="gpt-4o",
        max_loops=1,
    )

    chief_risk_officer = Agent(
        agent_name="Chief-Risk-Officer",
        agent_description="Chief Risk Officer with expertise in risk assessment and compliance",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Create comprehensive executive board
    executive_board = BoardOfDirectorsSwarm(
        name="Executive-Board-of-Directors",
        description="Executive board for high-level strategic decision making",
        agents=[
            chief_strategy_officer,
            chief_technology_officer,
            chief_financial_officer,
            chief_operating_officer,
            chief_risk_officer,
        ],
        max_loops=3,
        decision_threshold=0.8,  # Require strong consensus
        enable_voting=True,
        enable_consensus=True,
        verbose=True,
    )

    # Test with complex enterprise scenario
    result = executive_board.run(
        "Develop a comprehensive 5-year strategic plan for transforming our company into a "
        "leader in AI-powered enterprise solutions. Consider market opportunities, competitive "
        "landscape, technological requirements, financial implications, operational capabilities, "
        "and risk management strategies."
    )

    assert result is not None
