import pytest
from unittest.mock import Mock
from swarms.agents.multi_modal_agent import (
    MultiModalVisualAgent,
    MultiModalVisualAgentTool,
)


@pytest.fixture
def multimodal_agent():
    # Mock the MultiModalVisualAgent
    mock_agent = Mock(spec=MultiModalVisualAgent)
    mock_agent.run_text.return_value = "Expected output from agent"
    return mock_agent


@pytest.fixture
def multimodal_agent_tool(multimodal_agent):
    # Use the mocked MultiModalVisualAgent in the MultiModalVisualAgentTool
    return MultiModalVisualAgentTool(multimodal_agent)


@pytest.mark.parametrize(
    "text_input, expected_output",
    [
        ("Hello, world!", "Expected output from agent"),
        ("Another task", "Expected output from agent"),
    ],
)
def test_run(multimodal_agent_tool, text_input, expected_output):
    assert multimodal_agent_tool._run(text_input) == expected_output

    # You can also test if the MultiModalVisualAgent's run_text method was called with the right argument
    multimodal_agent_tool.agent.run_text.assert_called_with(text_input)
