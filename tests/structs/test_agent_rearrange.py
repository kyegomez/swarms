import pytest
from swarms import Agent, AgentRearrange


# ============================================================================
# Helper Functions
# ============================================================================


def create_sample_agents():
    """Create sample agents for testing."""
    return [
        Agent(
            agent_name="ResearchAgent",
            agent_description="Specializes in researching topics",
            system_prompt="You are a research specialist. Provide concise answers.",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=True,
            streaming_on=True,
        ),
        Agent(
            agent_name="WriterAgent",
            agent_description="Expert in writing content",
            system_prompt="You are a writing expert. Provide concise answers.",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=True,
            streaming_on=True,
        ),
        Agent(
            agent_name="ReviewerAgent",
            agent_description="Expert in reviewing content",
            system_prompt="You are a review expert. Provide concise answers.",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=True,
            streaming_on=True,
        ),
    ]


# ============================================================================
# Initialization Tests
# ============================================================================


def test_initialization():
    """Test AgentRearrange initialization."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent -> ReviewerAgent",
        max_loops=1,
        verbose=True,
    )

    assert len(agent_rearrange.agents) == 3
    assert (
        agent_rearrange.flow
        == "ResearchAgent -> WriterAgent -> ReviewerAgent"
    )
    assert agent_rearrange.max_loops == 1
    assert agent_rearrange.verbose is True
    print("✓ test_initialization passed")


def test_initialization_with_team_awareness():
    """Test AgentRearrange with team_awareness enabled."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent -> ReviewerAgent",
        team_awareness=True,
        verbose=True,
    )

    assert (
        agent_rearrange.flow
        == "ResearchAgent -> WriterAgent -> ReviewerAgent"
    )
    print("✓ test_initialization_with_team_awareness passed")


def test_initialization_with_custom_output_type():
    """Test AgentRearrange with custom output types."""
    agents = create_sample_agents()

    for output_type in ["all", "final", "list", "dict"]:
        agent_rearrange = AgentRearrange(
            agents=agents,
            flow="ResearchAgent -> WriterAgent",
            output_type=output_type,
            verbose=True,
        )
        assert agent_rearrange.output_type == output_type

    print("✓ test_initialization_with_custom_output_type passed")


# ============================================================================
# Agent Management Tests
# ============================================================================


def test_add_agent():
    """Test adding an agent to AgentRearrange."""
    agents = create_sample_agents()[:2]  # Only 2 agents

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent",
        verbose=True,
    )

    new_agent = Agent(
        agent_name="EditorAgent",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=True,
        streaming_on=True,
    )

    agent_rearrange.add_agent(new_agent)
    assert "EditorAgent" in agent_rearrange.agents
    assert len(agent_rearrange.agents) == 3
    print("✓ test_add_agent passed")


def test_remove_agent():
    """Test removing an agent from AgentRearrange."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent -> ReviewerAgent",
        verbose=True,
    )

    agent_rearrange.remove_agent("ReviewerAgent")
    assert "ReviewerAgent" not in agent_rearrange.agents
    assert len(agent_rearrange.agents) == 2
    print("✓ test_remove_agent passed")


def test_add_agents():
    """Test adding multiple agents to AgentRearrange."""
    agents = create_sample_agents()[:1]  # Start with 1 agent

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent",
        verbose=True,
    )

    new_agents = [
        Agent(
            agent_name="Agent4",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=True,
            streaming_on=True,
        ),
        Agent(
            agent_name="Agent5",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=True,
            streaming_on=True,
        ),
    ]

    agent_rearrange.add_agents(new_agents)
    assert "Agent4" in agent_rearrange.agents
    assert "Agent5" in agent_rearrange.agents
    assert len(agent_rearrange.agents) == 3
    print("✓ test_add_agents passed")


# ============================================================================
# Flow Validation Tests
# ============================================================================


def test_validate_flow_valid():
    """Test flow validation with valid flow."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent -> ReviewerAgent",
        verbose=True,
    )

    assert agent_rearrange.validate_flow() is True
    print("✓ test_validate_flow_valid passed")


def test_validate_flow_invalid():
    """Test flow validation with invalid agent name in flow."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent -> ReviewerAgent",
        verbose=True,
    )

    # Change to invalid flow
    agent_rearrange.flow = "ResearchAgent -> NonExistentAgent"

    try:
        agent_rearrange.validate_flow()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not registered" in str(e)
        print("✓ test_validate_flow_invalid passed")


def test_validate_flow_no_arrow():
    """Test flow validation without arrow syntax."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent",
        verbose=True,
    )

    agent_rearrange.flow = "ResearchAgent WriterAgent"

    try:
        agent_rearrange.validate_flow()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "'->" in str(e)
        print("✓ test_validate_flow_no_arrow passed")


# ============================================================================
# Flow Pattern Tests
# ============================================================================


def test_set_custom_flow():
    """Test setting custom flow."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent",
        verbose=True,
    )

    new_flow = "WriterAgent -> ResearchAgent -> ReviewerAgent"
    agent_rearrange.set_custom_flow(new_flow)
    assert agent_rearrange.flow == new_flow
    print("✓ test_set_custom_flow passed")


def test_sequential_flow():
    """Test sequential flow execution."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent",
        max_loops=1,
        verbose=True,
    )

    result = agent_rearrange.run("What is 2+2?")
    assert result is not None
    print("✓ test_sequential_flow passed")


def test_concurrent_flow():
    """Test concurrent flow execution with comma syntax."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent, WriterAgent -> ReviewerAgent",
        max_loops=1,
        verbose=True,
    )

    result = agent_rearrange.run("What is 3+3?")
    assert result is not None
    print("✓ test_concurrent_flow passed")


# ============================================================================
# Sequential Awareness Tests
# ============================================================================


def test_get_sequential_flow_structure():
    """Test getting sequential flow structure."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent -> ReviewerAgent",
        verbose=True,
    )

    flow_structure = agent_rearrange.get_sequential_flow_structure()
    assert flow_structure is not None
    assert isinstance(flow_structure, str)
    print("✓ test_get_sequential_flow_structure passed")


def test_get_agent_sequential_awareness():
    """Test getting agent sequential awareness."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent -> ReviewerAgent",
        verbose=True,
    )

    awareness = agent_rearrange.get_agent_sequential_awareness(
        "WriterAgent"
    )
    assert awareness is not None
    assert isinstance(awareness, str)
    print("✓ test_get_agent_sequential_awareness passed")


# ============================================================================
# Execution Tests
# ============================================================================


def test_run_basic():
    """Test basic run execution."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent",
        max_loops=1,
        verbose=True,
    )

    result = agent_rearrange.run("Calculate 5+5")
    assert result is not None
    print("✓ test_run_basic passed")


def test_run_with_different_output_types():
    """Test run with different output types."""
    agents = create_sample_agents()

    for output_type in ["all", "final"]:
        agent_rearrange = AgentRearrange(
            agents=agents,
            flow="ResearchAgent -> WriterAgent",
            output_type=output_type,
            max_loops=1,
            verbose=True,
        )

        result = agent_rearrange.run("What is the capital of France?")
        assert result is not None

    print("✓ test_run_with_different_output_types passed")


def test_callable_execution():
    """Test __call__ method."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent",
        max_loops=1,
        verbose=True,
    )

    result = agent_rearrange("What is 10+10?")
    assert result is not None
    print("✓ test_callable_execution passed")


# ============================================================================
# Batch Processing Tests
# ============================================================================


def test_batch_run():
    """Test batch processing."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent",
        max_loops=1,
        verbose=True,
    )

    tasks = ["What is 1+1?", "What is 2+2?"]
    results = agent_rearrange.batch_run(tasks, batch_size=2)

    assert results is not None
    assert len(results) == 2
    print("✓ test_batch_run passed")


def test_concurrent_run():
    """Test concurrent execution."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent",
        max_loops=1,
        verbose=True,
    )

    tasks = ["What is 3+3?", "What is 4+4?"]
    results = agent_rearrange.concurrent_run(tasks, max_workers=2)

    assert results is not None
    assert len(results) == 2
    print("✓ test_concurrent_run passed")


# ============================================================================
# Serialization Tests
# ============================================================================


def test_to_dict():
    """Test serialization to dictionary."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent",
        verbose=True,
    )

    result_dict = agent_rearrange.to_dict()
    assert isinstance(result_dict, dict)
    assert "name" in result_dict
    assert "flow" in result_dict
    assert "agents" in result_dict
    print("✓ test_to_dict passed")


# ============================================================================
# Integration Tests
# ============================================================================


def test_complete_workflow():
    """Test complete workflow with all features."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        name="test-workflow",
        description="Test complete workflow",
        agents=agents,
        flow="ResearchAgent -> WriterAgent -> ReviewerAgent",
        max_loops=1,
        team_awareness=True,
        verbose=True,
        output_type="all",
    )

    # Single run
    result1 = agent_rearrange.run("What is Python?")
    assert result1 is not None

    # Get flow structure
    flow_structure = agent_rearrange.get_sequential_flow_structure()
    assert flow_structure is not None

    # Serialize
    result_dict = agent_rearrange.to_dict()
    assert isinstance(result_dict, dict)

    print("✓ test_complete_workflow passed")


def main():
    """Run all tests."""
    tests = [
        test_initialization,
        test_initialization_with_team_awareness,
        test_initialization_with_custom_output_type,
        test_add_agent,
        test_remove_agent,
        test_add_agents,
        test_validate_flow_valid,
        test_validate_flow_invalid,
        test_validate_flow_no_arrow,
        test_set_custom_flow,
        test_sequential_flow,
        test_concurrent_flow,
        test_get_sequential_flow_structure,
        test_get_agent_sequential_awareness,
        test_run_basic,
        test_run_with_different_output_types,
        test_callable_execution,
        test_batch_run,
        test_concurrent_run,
        test_to_dict,
        test_complete_workflow,
    ]

    print("=" * 60)
    print("Running AgentRearrange Tests")
    print("=" * 60)

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
