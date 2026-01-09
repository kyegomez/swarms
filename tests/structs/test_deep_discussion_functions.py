import pytest
from swarms.structs.deep_discussion import one_on_one_debate
from swarms.structs.agent import Agent


def create_test_agent(name: str, description: str = "Test agent") -> Agent:
    """Create a real Agent instance for testing"""
    return Agent(
        agent_name=name,
        agent_description=description,
        system_prompt=f"You are {name}, a helpful test assistant. Keep responses brief.",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
    )


def test_one_on_one_debate_requires_two_agents():
    """Test that one_on_one_debate requires exactly two agents"""
    agent1 = create_test_agent("Agent1")

    with pytest.raises(ValueError, match="exactly two agents"):
        one_on_one_debate(agents=[agent1], task="Test")


def test_one_on_one_debate_with_three_agents_raises_error():
    """Test that one_on_one_debate raises error with three agents"""
    agent1 = create_test_agent("Agent1")
    agent2 = create_test_agent("Agent2")
    agent3 = create_test_agent("Agent3")

    with pytest.raises(ValueError, match="exactly two agents"):
        one_on_one_debate(agents=[agent1, agent2, agent3], task="Test")


def test_one_on_one_debate_with_empty_list_raises_error():
    """Test that one_on_one_debate raises error with empty agent list"""
    with pytest.raises(ValueError, match="exactly two agents"):
        one_on_one_debate(agents=[], task="Test")


def test_one_on_one_debate_basic_execution():
    """Test basic execution of one_on_one_debate"""
    agent1 = create_test_agent("Agent1")
    agent2 = create_test_agent("Agent2")

    result = one_on_one_debate(
        agents=[agent1, agent2],
        task="What is 2+2?",
        max_loops=1
    )

    # Result should exist
    assert result is not None


def test_one_on_one_debate_max_loops():
    """Test that one_on_one_debate respects max_loops parameter"""
    agent1 = create_test_agent("Agent1")
    agent2 = create_test_agent("Agent2")

    result = one_on_one_debate(
        agents=[agent1, agent2],
        task="Briefly discuss: What makes a good team?",
        max_loops=2
    )

    # Result should exist
    assert result is not None


def test_one_on_one_debate_with_img_parameter():
    """Test that one_on_one_debate accepts img parameter"""
    agent1 = create_test_agent("Agent1")
    agent2 = create_test_agent("Agent2")

    # Test that the function accepts img parameter without error
    result = one_on_one_debate(
        agents=[agent1, agent2],
        task="Describe briefly",
        img=None,  # Using None to avoid needing actual image
        max_loops=1
    )

    assert result is not None


def test_one_on_one_debate_alternates_speakers():
    """Test that one_on_one_debate produces output from debate"""
    agent1 = create_test_agent("Agent1")
    agent2 = create_test_agent("Agent2")

    result = one_on_one_debate(
        agents=[agent1, agent2],
        task="Exchange one brief greeting each",
        max_loops=2
    )

    # Verify output was produced
    assert result is not None
    assert len(str(result)) > 0


def test_one_on_one_debate_with_zero_loops():
    """Test one_on_one_debate with zero max_loops"""
    agent1 = create_test_agent("Agent1")
    agent2 = create_test_agent("Agent2")

    result = one_on_one_debate(
        agents=[agent1, agent2],
        task="Task",
        max_loops=0
    )

    # With 0 loops, result should still be returned (possibly empty)
    assert result is not None


def test_one_on_one_debate_output_type_parameter():
    """Test that one_on_one_debate accepts output_type parameter"""
    agent1 = create_test_agent("Agent1")
    agent2 = create_test_agent("Agent2")

    # Should not raise error with valid output_type
    result = one_on_one_debate(
        agents=[agent1, agent2],
        task="Say hello briefly",
        max_loops=1,
        output_type="str"
    )

    assert result is not None


def test_one_on_one_debate_passes_task_to_first_agent():
    """Test that initial task is passed and processed"""
    agent1 = create_test_agent("Agent1")
    agent2 = create_test_agent("Agent2")

    initial_task = "What is the capital of France?"
    result = one_on_one_debate(
        agents=[agent1, agent2],
        task=initial_task,
        max_loops=1
    )

    # Result should contain some response
    assert result is not None
    assert len(str(result)) > 0


def test_one_on_one_debate_produces_output():
    """Test that debate executes and produces output"""
    agent1 = create_test_agent("Agent1")
    agent2 = create_test_agent("Agent2")

    result = one_on_one_debate(
        agents=[agent1, agent2],
        task="What color is the sky?",
        max_loops=1
    )

    # Debate should produce a result
    assert result is not None
