import pytest
from swarms.structs.utils import find_agent_by_id, find_agent_by_name
from swarms.structs.agent import Agent


def create_test_agent(name: str, agent_id: str = None, description: str = "Test agent") -> Agent:
    """Create a real Agent instance for testing"""
    agent = Agent(
        agent_name=name,
        agent_description=description,
        system_prompt=f"You are {name}, a helpful test assistant. Keep responses brief.",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
    )
    if agent_id:
        agent.id = agent_id
    # Set name attribute for find_agent_by_name
    agent.name = name
    return agent


def test_find_agent_by_id_found():
    """Test finding agent by ID when agent exists"""
    agent1 = create_test_agent("Agent One", "agent-1")
    agent2 = create_test_agent("Agent Two", "agent-2")

    agents = [agent1, agent2]

    result = find_agent_by_id("agent-1", agents)
    assert result == agent1


def test_find_agent_by_id_not_found():
    """Test finding agent by ID when agent does not exist"""
    agent1 = create_test_agent("Agent One", "agent-1")

    agents = [agent1]

    result = find_agent_by_id("agent-99", agents)
    assert result is None


def test_find_agent_by_id_with_task():
    """Test finding agent by ID and running a task"""
    agent = create_test_agent("Agent One", "agent-1")

    agents = [agent]

    result = find_agent_by_id("agent-1", agents, task="What is 2+2?")
    assert result is not None
    assert len(str(result)) > 0


def test_find_agent_by_id_with_task_and_kwargs():
    """Test finding agent by ID and running a task with kwargs"""
    agent = create_test_agent("Agent One", "agent-1")

    agents = [agent]

    result = find_agent_by_id(
        agent_id="agent-1", agents=agents, task="Say hello"
    )
    assert result is not None


def test_find_agent_by_id_empty_list():
    """Test finding agent by ID in empty list"""
    result = find_agent_by_id("agent-1", [])
    assert result is None


def test_find_agent_by_id_exception_handling():
    """Test that find_agent_by_id handles task execution"""
    agent = create_test_agent("Agent One", "agent-1")

    agents = [agent]

    # Should execute successfully with real agent
    result = find_agent_by_id("agent-1", agents, task="What is the capital of France?")
    assert result is not None


def test_find_agent_by_name_found():
    """Test finding agent by name when agent exists"""
    agent1 = create_test_agent("Agent One", "agent-1")
    agent2 = create_test_agent("Agent Two", "agent-2")

    agents = [agent1, agent2]

    result = find_agent_by_name("Agent One", agents)
    assert result == agent1


def test_find_agent_by_name_not_found():
    """Test finding agent by name when agent does not exist"""
    agent1 = create_test_agent("Agent One", "agent-1")

    agents = [agent1]

    result = find_agent_by_name("Agent Ninety Nine", agents)
    assert result is None


def test_find_agent_by_name_with_task():
    """Test finding agent by name and running a task"""
    agent = create_test_agent("Agent One", "agent-1")

    agents = [agent]

    result = find_agent_by_name("Agent One", agents, task="What is 3+3?")
    assert result is not None
    assert len(str(result)) > 0


def test_find_agent_by_name_with_task_and_kwargs():
    """Test finding agent by name and running a task with kwargs"""
    agent = create_test_agent("Agent One", "agent-1")

    agents = [agent]

    result = find_agent_by_name(
        agent_name="Agent One", agents=agents, task="Say goodbye"
    )
    assert result is not None


def test_find_agent_by_name_empty_list():
    """Test finding agent by name in empty list"""
    result = find_agent_by_name("Agent One", [])
    assert result is None


def test_find_agent_by_name_exception_handling():
    """Test that find_agent_by_name handles task execution"""
    agent = create_test_agent("Agent One", "agent-1")

    agents = [agent]

    # Should execute successfully with real agent
    result = find_agent_by_name("Agent One", agents, task="List 3 colors")
    assert result is not None


def test_find_agent_by_id_multiple_agents():
    """Test finding correct agent by ID when multiple agents exist"""
    agents = []
    for i in range(10):
        agent = create_test_agent(f"Agent {i}", f"agent-{i}")
        agents.append(agent)

    result = find_agent_by_id("agent-5", agents)
    assert result.id == "agent-5"


def test_find_agent_by_name_multiple_agents():
    """Test finding correct agent by name when multiple agents exist"""
    agents = []
    for i in range(10):
        agent = create_test_agent(f"Agent {i}", f"agent-{i}")
        agents.append(agent)

    result = find_agent_by_name("Agent 5", agents)
    assert result.name == "Agent 5"
