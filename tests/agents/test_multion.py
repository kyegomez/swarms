import pytest
from unittest.mock import patch, MagicMock
from swarms.agents.multion_agent import MultiOnAgent


@patch("swarms.agents.multion_agent.multion")
def test_multion_agent_run(mock_multion):
    mock_response = MagicMock()
    mock_response.result = "result"
    mock_response.status = "status"
    mock_response.lastUrl = "lastUrl"
    mock_multion.browse.return_value = mock_response

    agent = MultiOnAgent(
        multion_api_key="test_key",
        max_steps=5,
        starting_url="https://www.example.com",
    )
    result, status, last_url = agent.run("task")

    assert result == "result"
    assert status == "status"
    assert last_url == "lastUrl"
    mock_multion.browse.assert_called_once_with(
        {
            "cmd": "task",
            "url": "https://www.example.com",
            "maxSteps": 5,
        }
    )


# Additional tests for different tasks
@pytest.mark.parametrize(
    "task", ["task1", "task2", "task3", "task4", "task5"]
)
@patch("swarms.agents.multion_agent.multion")
def test_multion_agent_run_different_tasks(mock_multion, task):
    mock_response = MagicMock()
    mock_response.result = "result"
    mock_response.status = "status"
    mock_response.lastUrl = "lastUrl"
    mock_multion.browse.return_value = mock_response

    agent = MultiOnAgent(
        multion_api_key="test_key",
        max_steps=5,
        starting_url="https://www.example.com",
    )
    result, status, last_url = agent.run(task)

    assert result == "result"
    assert status == "status"
    assert last_url == "lastUrl"
    mock_multion.browse.assert_called_once_with(
        {"cmd": task, "url": "https://www.example.com", "maxSteps": 5}
    )
