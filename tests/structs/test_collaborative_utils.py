import pytest
from swarms.structs.collaborative_utils import talk_to_agent
from swarms.structs.agent import Agent


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


def test_talk_to_agent_success():
    """Test successful agent-to-agent communication"""
    current_agent = create_test_agent("CurrentAgent")
    target_agent = create_test_agent("TargetAgent")

    agents = [current_agent, target_agent]

    result = talk_to_agent(
        current_agent=current_agent,
        agents=agents,
        task="What is 2+2?",
        agent_name="TargetAgent",
        max_loops=1
    )

    assert result is not None
    # Result should be a list or string from the debate
    assert len(str(result)) > 0


def test_talk_to_agent_not_found():
    """Test error when target agent not found"""
    current_agent = create_test_agent("CurrentAgent")

    agents = [current_agent]

    with pytest.raises(ValueError, match="Agent 'NonExistent' not found"):
        talk_to_agent(
            current_agent=current_agent,
            agents=agents,
            task="Test task",
            agent_name="NonExistent"
        )


def test_talk_to_agent_with_max_loops():
    """Test talk_to_agent with custom max_loops"""
    current_agent = create_test_agent("CurrentAgent")
    target_agent = create_test_agent("TargetAgent")

    agents = [current_agent, target_agent]

    result = talk_to_agent(
        current_agent=current_agent,
        agents=agents,
        task="Discuss the weather briefly",
        agent_name="TargetAgent",
        max_loops=2
    )

    assert result is not None


def test_talk_to_agent_no_agent_name_attribute():
    """Test when target agent is not found by name"""
    current_agent = create_test_agent("CurrentAgent")
    # Create another agent with different name
    other_agent = create_test_agent("OtherAgent")

    agents = [current_agent, other_agent]

    with pytest.raises(ValueError, match="Agent 'TargetAgent' not found"):
        talk_to_agent(
            current_agent=current_agent,
            agents=agents,
            task="Test",
            agent_name="TargetAgent"
        )


def test_talk_to_agent_output_type():
    """Test talk_to_agent with custom output_type"""
    current_agent = create_test_agent("CurrentAgent")
    target_agent = create_test_agent("TargetAgent")

    agents = [current_agent, target_agent]

    result = talk_to_agent(
        current_agent=current_agent,
        agents=agents,
        task="Say hello",
        agent_name="TargetAgent",
        output_type="str",
        max_loops=1
    )

    assert result is not None
