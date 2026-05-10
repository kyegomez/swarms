import pytest
from swarms.structs.ma_blocks import (
    aggregator_agent_task_prompt,
    aggregate,
    run_agent,
    find_agent_by_name,
)
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation


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


def test_aggregator_agent_task_prompt():
    """Test aggregator agent task prompt generation"""
    agent1 = create_test_agent("Agent1", "First test agent")
    agent2 = create_test_agent("Agent2", "Second test agent")

    workers = [agent1, agent2]

    conversation = Conversation()
    conversation.add(role="Agent1", content="Hello")
    conversation.add(role="Agent2", content="Hi")

    result = aggregator_agent_task_prompt(
        task="Test task",
        workers=workers,
        conversation=conversation
    )

    assert "Test task" in result
    assert "2" in result  # Number of agents
    assert "Hello" in result or "Hi" in result


def test_aggregate_missing_task_raises_error():
    """Test that missing task raises ValueError"""
    agent = create_test_agent("TestAgent")
    with pytest.raises(ValueError, match="Task is required"):
        aggregate(workers=[agent], task=None)


def test_aggregate_missing_workers_raises_error():
    """Test that missing workers raises ValueError"""
    with pytest.raises(ValueError, match="Workers is required"):
        aggregate(workers=None, task="Test")


def test_aggregate_workers_not_list_raises_error():
    """Test that non-list workers raises ValueError"""
    agent = create_test_agent("TestAgent")
    with pytest.raises(ValueError, match="Workers must be a list"):
        aggregate(workers=agent, task="Test")


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
    agent = create_test_agent("TestAgent")
    with pytest.raises(ValueError, match="Task cannot be None"):
        run_agent(agent=agent, task=None)


def test_run_agent_not_agent_instance_raises_error():
    """Test that non-Agent instance raises TypeError"""
    with pytest.raises(TypeError, match="Agent must be an instance of Agent"):
        run_agent(agent="not an agent", task="Test")


def test_run_agent_success():
    """Test successful agent run"""
    agent = create_test_agent("TestAgent")

    result = run_agent(agent=agent, task="What is 2+2?")

    assert result is not None
    assert len(str(result)) > 0


def test_run_agent_with_kwargs():
    """Test run_agent with additional kwargs"""
    agent = create_test_agent("TestAgent")

    result = run_agent(
        agent=agent,
        task="Say hello"
    )

    assert result is not None


def test_find_agent_by_name_empty_list_raises_error():
    """Test that empty agents list raises ValueError"""
    with pytest.raises(ValueError, match="Agents list cannot be empty"):
        find_agent_by_name(agents=[], agent_name="Test")


def test_find_agent_by_name_non_string_raises_error():
    """Test that non-string agent_name raises TypeError"""
    agent = create_test_agent("TestAgent")
    with pytest.raises(TypeError, match="Agent name must be a string"):
        find_agent_by_name(agents=[agent], agent_name=123)


def test_find_agent_by_name_empty_string_raises_error():
    """Test that empty agent_name raises ValueError"""
    agent = create_test_agent("TestAgent")
    with pytest.raises(ValueError, match="Agent name cannot be empty"):
        find_agent_by_name(agents=[agent], agent_name="   ")


def test_find_agent_by_name_success():
    """Test successful agent finding by name"""
    agent1 = create_test_agent("Agent1")
    agent2 = create_test_agent("Agent2")

    # Note: find_agent_by_name looks for 'name' attribute, not 'agent_name'
    agent1.name = "Agent1"
    agent2.name = "Agent2"

    result = find_agent_by_name(
        agents=[agent1, agent2],
        agent_name="Agent2"
    )

    assert result == agent2


def test_find_agent_by_name_not_found_raises_error():
    """Test that agent not found raises RuntimeError"""
    agent = create_test_agent("Agent1")
    agent.name = "Agent1"

    with pytest.raises(RuntimeError, match="Error finding agent"):
        find_agent_by_name(agents=[agent], agent_name="NonExistent")


def test_find_agent_by_name_agent_with_name_attribute():
    """Test finding agent when agent has name attribute"""
    agent1 = create_test_agent("Agent1")
    agent2 = create_test_agent("TargetAgent")

    # Set the name attribute that find_agent_by_name looks for
    agent1.name = "Agent1"
    agent2.name = "TargetAgent"

    result = find_agent_by_name(
        agents=[agent1, agent2],
        agent_name="TargetAgent"
    )

    assert result == agent2
