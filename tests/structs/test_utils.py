import pytest
from unittest.mock import Mock
from swarms.structs.utils import find_agent_by_id, find_agent_by_name


def test_find_agent_by_id_found():
    """Test finding agent by ID when agent exists"""
    mock_agent1 = Mock()
    mock_agent1.id = "agent-1"
    mock_agent1.name = "Agent One"

    mock_agent2 = Mock()
    mock_agent2.id = "agent-2"
    mock_agent2.name = "Agent Two"

    agents = [mock_agent1, mock_agent2]

    result = find_agent_by_id("agent-1", agents)
    assert result == mock_agent1


def test_find_agent_by_id_not_found():
    """Test finding agent by ID when agent does not exist"""
    mock_agent1 = Mock()
    mock_agent1.id = "agent-1"

    agents = [mock_agent1]

    result = find_agent_by_id("agent-99", agents)
    assert result is None


def test_find_agent_by_id_with_task():
    """Test finding agent by ID and running a task"""
    mock_agent = Mock()
    mock_agent.id = "agent-1"
    mock_agent.run.return_value = "Task completed"

    agents = [mock_agent]

    result = find_agent_by_id("agent-1", agents, task="Do something")
    assert result == "Task completed"
    mock_agent.run.assert_called_once_with("Do something")


def test_find_agent_by_id_with_task_and_args():
    """Test finding agent by ID and running a task with args and kwargs"""
    mock_agent = Mock()
    mock_agent.id = "agent-1"
    mock_agent.run.return_value = "Task completed"

    agents = [mock_agent]

    result = find_agent_by_id(
        agent_id="agent-1", agents=agents, task="Do something", kwarg1="value1"
    )
    assert result == "Task completed"
    mock_agent.run.assert_called_once_with("Do something", kwarg1="value1")


def test_find_agent_by_id_empty_list():
    """Test finding agent by ID in empty list"""
    result = find_agent_by_id("agent-1", [])
    assert result is None


def test_find_agent_by_id_exception_handling():
    """Test that find_agent_by_id handles exceptions gracefully"""
    mock_agent = Mock()
    mock_agent.id = "agent-1"
    mock_agent.run.side_effect = Exception("Test error")

    agents = [mock_agent]

    result = find_agent_by_id("agent-1", agents, task="Do something")
    assert result is None


def test_find_agent_by_name_found():
    """Test finding agent by name when agent exists"""
    mock_agent1 = Mock()
    mock_agent1.id = "agent-1"
    mock_agent1.name = "Agent One"

    mock_agent2 = Mock()
    mock_agent2.id = "agent-2"
    mock_agent2.name = "Agent Two"

    agents = [mock_agent1, mock_agent2]

    result = find_agent_by_name("Agent One", agents)
    assert result == mock_agent1


def test_find_agent_by_name_not_found():
    """Test finding agent by name when agent does not exist"""
    mock_agent1 = Mock()
    mock_agent1.name = "Agent One"

    agents = [mock_agent1]

    result = find_agent_by_name("Agent Ninety Nine", agents)
    assert result is None


def test_find_agent_by_name_with_task():
    """Test finding agent by name and running a task"""
    mock_agent = Mock()
    mock_agent.name = "Agent One"
    mock_agent.run.return_value = "Task completed"

    agents = [mock_agent]

    result = find_agent_by_name("Agent One", agents, task="Do something")
    assert result == "Task completed"
    mock_agent.run.assert_called_once_with("Do something")


def test_find_agent_by_name_with_task_and_args():
    """Test finding agent by name and running a task with args and kwargs"""
    mock_agent = Mock()
    mock_agent.name = "Agent One"
    mock_agent.run.return_value = "Task completed"

    agents = [mock_agent]

    result = find_agent_by_name(
        agent_name="Agent One", agents=agents, task="Do something", kwarg1="value1"
    )
    assert result == "Task completed"
    mock_agent.run.assert_called_once_with("Do something", kwarg1="value1")


def test_find_agent_by_name_empty_list():
    """Test finding agent by name in empty list"""
    result = find_agent_by_name("Agent One", [])
    assert result is None


def test_find_agent_by_name_exception_handling():
    """Test that find_agent_by_name handles exceptions gracefully"""
    mock_agent = Mock()
    mock_agent.name = "Agent One"
    mock_agent.run.side_effect = Exception("Test error")

    agents = [mock_agent]

    result = find_agent_by_name("Agent One", agents, task="Do something")
    assert result is None


def test_find_agent_by_id_multiple_agents():
    """Test finding correct agent by ID when multiple agents exist"""
    agents = []
    for i in range(10):
        agent = Mock()
        agent.id = f"agent-{i}"
        agents.append(agent)

    result = find_agent_by_id("agent-5", agents)
    assert result.id == "agent-5"


def test_find_agent_by_name_multiple_agents():
    """Test finding correct agent by name when multiple agents exist"""
    agents = []
    for i in range(10):
        agent = Mock()
        agent.name = f"Agent {i}"
        agents.append(agent)

    result = find_agent_by_name("Agent 5", agents)
    assert result.name == "Agent 5"
