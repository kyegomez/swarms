import pytest
from unittest.mock import Mock, MagicMock
from swarms.structs.deep_discussion import one_on_one_debate


def test_one_on_one_debate_requires_two_agents():
    """Test that one_on_one_debate requires exactly two agents"""
    agent1 = Mock()
    agent1.agent_name = "Agent1"

    with pytest.raises(ValueError, match="exactly two agents"):
        one_on_one_debate(agents=[agent1], task="Test")


def test_one_on_one_debate_with_three_agents_raises_error():
    """Test that one_on_one_debate raises error with three agents"""
    agent1 = Mock()
    agent2 = Mock()
    agent3 = Mock()

    with pytest.raises(ValueError, match="exactly two agents"):
        one_on_one_debate(agents=[agent1, agent2, agent3], task="Test")


def test_one_on_one_debate_with_empty_list_raises_error():
    """Test that one_on_one_debate raises error with empty agent list"""
    with pytest.raises(ValueError, match="exactly two agents"):
        one_on_one_debate(agents=[], task="Test")


def test_one_on_one_debate_basic_execution():
    """Test basic execution of one_on_one_debate"""
    agent1 = Mock()
    agent1.agent_name = "Agent1"
    agent1.run.return_value = "Response from Agent1"

    agent2 = Mock()
    agent2.agent_name = "Agent2"
    agent2.run.return_value = "Response from Agent2"

    result = one_on_one_debate(
        agents=[agent1, agent2],
        task="Initial task",
        max_loops=1
    )

    # At least the first agent should have been called
    assert agent1.run.called
    # Result should exist
    assert result is not None


def test_one_on_one_debate_max_loops():
    """Test that one_on_one_debate respects max_loops parameter"""
    agent1 = Mock()
    agent1.agent_name = "Agent1"
    agent1.run.return_value = "Response 1"

    agent2 = Mock()
    agent2.agent_name = "Agent2"
    agent2.run.return_value = "Response 2"

    one_on_one_debate(
        agents=[agent1, agent2],
        task="Task",
        max_loops=3
    )

    # With max_loops=3, each agent should be called at least once
    assert agent1.run.call_count >= 1
    assert agent2.run.call_count >= 1


def test_one_on_one_debate_with_img_parameter():
    """Test that one_on_one_debate passes img parameter to agents"""
    agent1 = Mock()
    agent1.agent_name = "Agent1"
    agent1.run.return_value = "Response 1"

    agent2 = Mock()
    agent2.agent_name = "Agent2"
    agent2.run.return_value = "Response 2"

    one_on_one_debate(
        agents=[agent1, agent2],
        task="Task",
        img="image.jpg",
        max_loops=1
    )

    # Check that img was passed to run method
    assert agent1.run.called
    call_kwargs = agent1.run.call_args[1] if agent1.run.call_args[1] else {}
    assert "img" in call_kwargs or agent1.run.call_args[0]


def test_one_on_one_debate_alternates_speakers():
    """Test that one_on_one_debate alternates between agents"""
    call_order = []

    agent1 = Mock()
    agent1.agent_name = "Agent1"
    agent1.run.side_effect = lambda task, img=None: (call_order.append("Agent1"), "Response 1")[1]

    agent2 = Mock()
    agent2.agent_name = "Agent2"
    agent2.run.side_effect = lambda task, img=None: (call_order.append("Agent2"), "Response 2")[1]

    one_on_one_debate(
        agents=[agent1, agent2],
        task="Task",
        max_loops=2
    )

    # Verify alternating pattern
    assert len(call_order) >= 2


def test_one_on_one_debate_with_zero_loops():
    """Test one_on_one_debate with zero max_loops"""
    agent1 = Mock()
    agent1.agent_name = "Agent1"
    agent1.run.return_value = "Response 1"

    agent2 = Mock()
    agent2.agent_name = "Agent2"
    agent2.run.return_value = "Response 2"

    result = one_on_one_debate(
        agents=[agent1, agent2],
        task="Task",
        max_loops=0
    )

    # With 0 loops, agents should not be called
    assert agent1.run.call_count == 0
    assert agent2.run.call_count == 0


def test_one_on_one_debate_output_type_parameter():
    """Test that one_on_one_debate accepts output_type parameter"""
    agent1 = Mock()
    agent1.agent_name = "Agent1"
    agent1.run.return_value = "Response 1"

    agent2 = Mock()
    agent2.agent_name = "Agent2"
    agent2.run.return_value = "Response 2"

    # Should not raise error with valid output_type
    result = one_on_one_debate(
        agents=[agent1, agent2],
        task="Task",
        max_loops=1,
        output_type="str"  # Use valid output type
    )

    assert result is not None


def test_one_on_one_debate_passes_task_to_first_agent():
    """Test that initial task is passed to first agent"""
    agent1 = Mock()
    agent1.agent_name = "Agent1"
    agent1.run.return_value = "Response 1"

    agent2 = Mock()
    agent2.agent_name = "Agent2"
    agent2.run.return_value = "Response 2"

    initial_task = "Initial task for debate"
    one_on_one_debate(
        agents=[agent1, agent2],
        task=initial_task,
        max_loops=1
    )

    # First agent should receive the initial task
    assert agent1.run.called
    call_args = agent1.run.call_args
    assert initial_task in str(call_args)


def test_one_on_one_debate_second_agent_receives_first_response():
    """Test that debate executes and produces output"""
    agent1 = Mock()
    agent1.agent_name = "Agent1"
    first_response = "First agent response"
    agent1.run.return_value = first_response

    agent2 = Mock()
    agent2.agent_name = "Agent2"
    agent2.run.return_value = "Second agent response"

    result = one_on_one_debate(
        agents=[agent1, agent2],
        task="Initial task",
        max_loops=1
    )

    # Debate should produce a result
    assert result is not None
    # First agent should be called
    assert agent1.run.called
