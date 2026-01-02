from swarms.agents.i_agent import IterativeReflectiveExpansion


def test_ire_agent_initialization():
    """Test IRE agent initialization with default parameters"""
    agent = IterativeReflectiveExpansion()

    assert agent is not None
    assert agent.agent_name == "General-Reasoning-Agent"
    assert agent.max_loops == 5
    assert agent.output_type == "dict"
    assert agent.agent is not None


def test_ire_agent_custom_initialization():
    """Test IRE agent initialization with custom parameters"""
    agent = IterativeReflectiveExpansion(
        agent_name="Custom-IRE-Agent",
        description="A custom reasoning agent",
        max_loops=3,
        model_name="gpt-4o",
        output_type="string",
    )

    assert agent.agent_name == "Custom-IRE-Agent"
    assert agent.description == "A custom reasoning agent"
    assert agent.max_loops == 3
    assert agent.output_type == "string"


def test_ire_agent_execution():
    """Test IRE agent execution with a simple problem"""
    agent = IterativeReflectiveExpansion(
        agent_name="Test-IRE-Agent",
        model_name="gpt-4o",
        max_loops=2,
        output_type="dict",
    )

    # Test with a simple reasoning task
    task = "What are three main benefits of renewable energy?"
    result = agent.run(task)

    # Result should not be None
    assert result is not None
    # Result should be dict or string based on output_type
    assert isinstance(result, (str, dict))


def test_ire_agent_generate_hypotheses():
    """Test IRE agent hypothesis generation"""
    agent = IterativeReflectiveExpansion(
        agent_name="Hypothesis-Test-Agent",
        max_loops=1,
    )

    task = "How can we reduce carbon emissions?"
    hypotheses = agent.generate_initial_hypotheses(task)

    assert hypotheses is not None
    assert isinstance(hypotheses, list)
    assert len(hypotheses) > 0


def test_ire_agent_workflow():
    """Test complete IRE agent workflow with iterative refinement"""
    agent = IterativeReflectiveExpansion(
        agent_name="Workflow-Test-Agent",
        description="Agent for testing complete workflow",
        model_name="gpt-4o",
        max_loops=2,
        output_type="dict",
    )

    # Test with a problem that requires iterative refinement
    task = "Design an efficient public transportation system for a small city"
    result = agent.run(task)

    # Verify the result is valid
    assert result is not None
    assert isinstance(result, (str, dict))

    # Check that conversation was populated during execution
    assert agent.conversation is not None
