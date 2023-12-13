import os
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from dotenv import load_dotenv

from swarms.models.autotemp import AutoTempAgent

api_key = os.getenv("OPENAI_API_KEY")

load_dotenv()


@pytest.fixture
def auto_temp_agent():
    return AutoTempAgent(api_key=api_key)


def test_initialization(auto_temp_agent):
    assert isinstance(auto_temp_agent, AutoTempAgent)
    assert auto_temp_agent.auto_select is True
    assert auto_temp_agent.max_workers == 6
    assert auto_temp_agent.temperature == 0.5
    assert auto_temp_agent.alt_temps == [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]


def test_evaluate_output(auto_temp_agent):
    output = "This is a test output."
    with patch("swarms.models.OpenAIChat") as MockOpenAIChat:
        mock_instance = MockOpenAIChat.return_value
        mock_instance.return_value = "Score: 95.5"
        score = auto_temp_agent.evaluate_output(output)
        assert score == 95.5
        mock_instance.assert_called_once()


def test_run_auto_select(auto_temp_agent):
    task = "Generate a blog post."
    temperature_string = "0.4,0.6,0.8,1.0,1.2,1.4"
    result = auto_temp_agent.run(task, temperature_string)
    assert "Best AutoTemp Output" in result
    assert "Temp" in result
    assert "Score" in result


def test_run_no_scores(auto_temp_agent):
    task = "Invalid task."
    temperature_string = "0.4,0.6,0.8,1.0,1.2,1.4"
    with ThreadPoolExecutor(
        max_workers=auto_temp_agent.max_workers
    ) as executor:
        with patch.object(
            executor,
            "submit",
            side_effect=[None, None, None, None, None, None],
        ):
            result = auto_temp_agent.run(task, temperature_string)
            assert result == "No valid outputs generated."


def test_run_manual_select(auto_temp_agent):
    auto_temp_agent.auto_select = False
    task = "Generate a blog post."
    temperature_string = "0.4,0.6,0.8,1.0,1.2,1.4"
    result = auto_temp_agent.run(task, temperature_string)
    assert "Best AutoTemp Output" not in result
    assert "Temp" in result
    assert "Score" in result


def test_failed_initialization():
    with pytest.raises(Exception):
        AutoTempAgent()


def test_failed_evaluate_output(auto_temp_agent):
    output = "This is a test output."
    with patch("swarms.models.OpenAIChat") as MockOpenAIChat:
        mock_instance = MockOpenAIChat.return_value
        mock_instance.return_value = "Invalid score text"
        score = auto_temp_agent.evaluate_output(output)
        assert score == 0.0
