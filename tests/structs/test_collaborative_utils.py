import pytest
from unittest.mock import Mock, patch
from swarms.structs.collaborative_utils import talk_to_agent


def test_talk_to_agent_success():
    """Test successful agent-to-agent communication"""
    current_agent = Mock()
    current_agent.agent_name = "CurrentAgent"

    target_agent = Mock()
    target_agent.agent_name = "TargetAgent"

    agents = [current_agent, target_agent]

    # Mock the one_on_one_debate function result
    with patch('swarms.structs.collaborative_utils.one_on_one_debate') as mock_debate:
        mock_debate.return_value = ["conversation result"]

        result = talk_to_agent(
            current_agent=current_agent,
            agents=agents,
            task="Test task",
            agent_name="TargetAgent"
        )

        assert result == ["conversation result"]
        mock_debate.assert_called_once()


def test_talk_to_agent_not_found():
    """Test error when target agent not found"""
    current_agent = Mock()
    current_agent.agent_name = "CurrentAgent"

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
    current_agent = Mock()
    current_agent.agent_name = "CurrentAgent"

    target_agent = Mock()
    target_agent.agent_name = "TargetAgent"

    agents = [current_agent, target_agent]

    with patch('swarms.structs.collaborative_utils.one_on_one_debate') as mock_debate:
        mock_debate.return_value = ["result"]

        talk_to_agent(
            current_agent=current_agent,
            agents=agents,
            task="Test",
            agent_name="TargetAgent",
            max_loops=5
        )

        mock_debate.assert_called_once_with(
            max_loops=5,
            agents=[current_agent, target_agent],
            task="Test",
            output_type="str-all-except-first"
        )


def test_talk_to_agent_no_agent_name_attribute():
    """Test when agents don't have agent_name attribute"""
    current_agent = Mock()
    current_agent.agent_name = "CurrentAgent"

    target_agent = Mock(spec=[])  # No agent_name

    agents = [current_agent, target_agent]

    with pytest.raises(ValueError, match="Agent 'TargetAgent' not found"):
        talk_to_agent(
            current_agent=current_agent,
            agents=agents,
            task="Test",
            agent_name="TargetAgent"
        )


def test_talk_to_agent_output_type():
    """Test talk_to_agent with custom output_type"""
    current_agent = Mock()
    current_agent.agent_name = "CurrentAgent"

    target_agent = Mock()
    target_agent.agent_name = "TargetAgent"

    agents = [current_agent, target_agent]

    with patch('swarms.structs.collaborative_utils.one_on_one_debate') as mock_debate:
        mock_debate.return_value = ["result"]

        talk_to_agent(
            current_agent=current_agent,
            agents=agents,
            task="Test",
            agent_name="TargetAgent",
            output_type="custom"
        )

        args, kwargs = mock_debate.call_args
        assert kwargs["output_type"] == "custom"
