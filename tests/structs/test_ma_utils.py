import pytest
from swarms.structs.ma_utils import (
    list_all_agents,
    set_random_models_for_agents,
    create_agent_map,
)
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation


def create_test_agent(name: str, description: str = "Test agent") -> Agent:
    """Create a real Agent instance for testing"""
    return Agent(
        agent_name=name,
        agent_description=description,
        system_prompt=f"You are {name}, a helpful test assistant.",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
    )


def test_list_all_agents_basic():
    """Test basic listing of agents"""
    agent1 = create_test_agent("Agent1", "First agent")
    agent2 = create_test_agent("Agent2", "Second agent")

    result = list_all_agents([agent1, agent2], name="Test Team")

    assert "Test Team" in result
    assert "Total Agents: 2" in result
    assert "Agent1" in result
    assert "Agent2" in result


def test_list_all_agents_with_description():
    """Test listing agents with team description"""
    agent = create_test_agent("TestAgent", "Test description")

    result = list_all_agents(
        [agent],
        name="My Team",
        description="A great team"
    )

    assert "My Team" in result
    assert "A great team" in result


def test_list_all_agents_with_conversation():
    """Test adding agents to conversation"""
    agent = create_test_agent("Agent", "Desc")

    conversation = Conversation()

    result = list_all_agents(
        [agent],
        conversation=conversation,
        add_to_conversation=True
    )

    assert result is None
    # Conversation should have content added
    assert len(conversation.conversation_history) > 0


def test_list_all_agents_fallback_to_name():
    """Test that agent name uses agent_name attribute"""
    agent = create_test_agent("TestName", "Test description")

    result = list_all_agents([agent])
    assert "TestName" in result


def test_list_all_agents_fallback_to_system_prompt():
    """Test that description uses agent_description"""
    agent = create_test_agent("Agent", "Agent description here")

    result = list_all_agents([agent])
    assert "Agent" in result


def test_set_random_models_for_agents_with_none():
    """Test setting random model when agents is None"""
    result = set_random_models_for_agents(agents=None)
    assert isinstance(result, str)
    assert len(result) > 0


def test_set_random_models_for_agents_with_list():
    """Test setting random models for list of agents"""
    agent1 = create_test_agent("Agent1")
    agent2 = create_test_agent("Agent2")
    agents = [agent1, agent2]

    result = set_random_models_for_agents(agents=agents)

    assert result == agents
    assert hasattr(agent1, 'model_name')
    assert hasattr(agent2, 'model_name')


def test_set_random_models_for_agents_with_single_agent():
    """Test setting random model for single agent"""
    agent = create_test_agent("SingleAgent")

    result = set_random_models_for_agents(agents=agent)

    assert result == agent
    assert hasattr(agent, 'model_name')


def test_set_random_models_for_agents_custom_models():
    """Test setting random models with custom model list"""
    agent = create_test_agent("CustomAgent")
    custom_models = ["model1", "model2", "model3"]

    result = set_random_models_for_agents(agents=agent, model_names=custom_models)

    assert hasattr(agent, 'model_name')
    assert agent.model_name in custom_models


def test_create_agent_map_basic():
    """Test creating agent map with basic agents"""
    agent1 = create_test_agent("Agent1")
    agent2 = create_test_agent("Agent2")

    result = create_agent_map([agent1, agent2])

    assert "Agent1" in result
    assert "Agent2" in result
    assert result["Agent1"] == agent1
    assert result["Agent2"] == agent2


def test_create_agent_map_with_real_agents():
    """Test creating agent map with real Agent instances"""
    agent1 = create_test_agent("RealAgent1")
    agent2 = create_test_agent("RealAgent2")

    result = create_agent_map([agent1, agent2])

    assert "RealAgent1" in result
    assert "RealAgent2" in result
    assert result["RealAgent1"] == agent1
    assert result["RealAgent2"] == agent2


def test_create_agent_map_empty_raises_error():
    """Test that empty agent list raises ValueError"""
    with pytest.raises(ValueError, match="Agents list cannot be empty"):
        create_agent_map([])


def test_create_agent_map_caching():
    """Test that agent map is cached for identical inputs"""
    agent = create_test_agent("CachedAgent")

    agents = [agent]
    result1 = create_agent_map(agents)
    result2 = create_agent_map(agents)

    # Should return the same cached result
    assert result1 == result2


def test_list_all_agents_no_collaboration_prompt():
    """Test list_all_agents without collaboration prompt"""
    agent = create_test_agent("Agent", "Description")

    result = list_all_agents([agent], add_collaboration_prompt=False)

    assert "Agent" in result
