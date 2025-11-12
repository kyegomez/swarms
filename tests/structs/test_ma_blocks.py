import pytest
from unittest.mock import Mock, patch, MagicMock
from swarms.structs.ma_blocks import (
    aggregator_agent_task_prompt,
    aggregate,
    run_agent,
    find_agent_by_name,
)
from swarms.structs.agent import Agent


def test_aggregator_agent_task_prompt():
    """Test aggregator agent task prompt generation"""
    mock_agent1 = Mock()
    mock_agent1.agent_name = "Agent1"

    mock_agent2 = Mock()
    mock_agent2.agent_name = "Agent2"

    workers = [mock_agent1, mock_agent2]

    mock_conversation = Mock()
    mock_conversation.get_str.return_value = "Agent1: Hello\nAgent2: Hi"

    result = aggregator_agent_task_prompt(
        task="Test task",
        workers=workers,
        conversation=mock_conversation
    )

    assert "Test task" in result
    assert "2" in result  # Number of agents
    assert "Agent1: Hello" in result


def test_aggregate_missing_task_raises_error():
    """Test that missing task raises ValueError"""
    with pytest.raises(ValueError, match="Task is required"):
        aggregate(workers=[Mock()], task=None)


def test_aggregate_missing_workers_raises_error():
    """Test that missing workers raises ValueError"""
    with pytest.raises(ValueError, match="Workers is required"):
        aggregate(workers=None, task="Test")


def test_aggregate_workers_not_list_raises_error():
    """Test that non-list workers raises ValueError"""
    with pytest.raises(ValueError, match="Workers must be a list"):
        aggregate(workers=Mock(), task="Test")


def test_aggregate_workers_not_callable_raises_error():
    """Test that non-callable workers raises ValueError"""
    with pytest.raises(ValueError, match="Workers must be a list of Callable"):
        aggregate(workers=["not", "callable"], task="Test")


def test_run_agent_none_agent_raises_error():
    """Test that None agent raises ValueError"""
    with pytest.raises(ValueError, match="Agent cannot be None"):
        run_agent(agent=None, task="Test")


def test_run_agent_none_task_raises_error():
    """Test that None task raises ValueError"""
    mock_agent = Mock(spec=Agent)
    with pytest.raises(ValueError, match="Task cannot be None"):
        run_agent(agent=mock_agent, task=None)


def test_run_agent_not_agent_instance_raises_error():
    """Test that non-Agent instance raises TypeError"""
    with pytest.raises(TypeError, match="Agent must be an instance of Agent"):
        run_agent(agent="not an agent", task="Test")


def test_run_agent_success():
    """Test successful agent run"""
    mock_agent = Mock(spec=Agent)
    mock_agent.run.return_value = "Task completed"

    result = run_agent(agent=mock_agent, task="Test task")

    assert result == "Task completed"
    mock_agent.run.assert_called_once_with(task="Test task")


def test_run_agent_with_args_kwargs():
    """Test run_agent with additional args and kwargs"""
    mock_agent = Mock(spec=Agent)
    mock_agent.run.return_value = "Success"

    result = run_agent(
        agent=mock_agent,
        task="Test",
        extra_param="value"
    )

    assert result == "Success"
    mock_agent.run.assert_called_once_with(
        task="Test",
        extra_param="value"
    )


def test_run_agent_runtime_error_on_exception():
    """Test that exceptions during run raise RuntimeError"""
    mock_agent = Mock(spec=Agent)
    mock_agent.run.side_effect = Exception("Agent failed")

    with pytest.raises(RuntimeError, match="Error running agent"):
        run_agent(agent=mock_agent, task="Test")


def test_find_agent_by_name_empty_list_raises_error():
    """Test that empty agents list raises ValueError"""
    with pytest.raises(ValueError, match="Agents list cannot be empty"):
        find_agent_by_name(agents=[], agent_name="Test")


def test_find_agent_by_name_non_string_raises_error():
    """Test that non-string agent_name raises TypeError"""
    mock_agent = Mock()
    with pytest.raises(TypeError, match="Agent name must be a string"):
        find_agent_by_name(agents=[mock_agent], agent_name=123)


def test_find_agent_by_name_empty_string_raises_error():
    """Test that empty agent_name raises ValueError"""
    mock_agent = Mock()
    with pytest.raises(ValueError, match="Agent name cannot be empty"):
        find_agent_by_name(agents=[mock_agent], agent_name="   ")


def test_find_agent_by_name_success():
    """Test successful agent finding by name"""
    mock_agent1 = Mock()
    mock_agent1.name = "Agent1"

    mock_agent2 = Mock()
    mock_agent2.name = "Agent2"

    result = find_agent_by_name(
        agents=[mock_agent1, mock_agent2],
        agent_name="Agent2"
    )

    assert result == mock_agent2


def test_find_agent_by_name_not_found_raises_error():
    """Test that agent not found raises RuntimeError"""
    mock_agent = Mock()
    mock_agent.name = "Agent1"

    with pytest.raises(RuntimeError, match="Error finding agent"):
        find_agent_by_name(agents=[mock_agent], agent_name="NonExistent")


def test_find_agent_by_name_agent_without_name_attribute():
    """Test finding agent when some agents don't have name attribute"""
    mock_agent1 = Mock(spec=[])  # No name attribute
    mock_agent2 = Mock()
    mock_agent2.name = "TargetAgent"

    result = find_agent_by_name(
        agents=[mock_agent1, mock_agent2],
        agent_name="TargetAgent"
    )

    assert result == mock_agent2
