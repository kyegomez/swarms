from unittest.mock import MagicMock, patch
import pytest
from swarms.structs.agent import Agent


def test_agent_run_warns_on_length_cutoff_empty_output():
    agent = Agent(
        agent_name="TestAgent",
        model_name="openai/gpt-4o",
        max_loops=1,
        max_tokens=64,
        print_on=False,
        persistent_memory=False,
    )

    mock_llm = MagicMock()
    mock_llm.last_finish_reason = "length"
    mock_llm.last_response_model = "gpt-4o"
    mock_llm.last_response = None
    mock_llm.stream = False
    mock_llm.usage = {
        "input_tokens": 10,
        "output_tokens": 64,
        "cached_tokens": 0,
        "reasoning_tokens": 64,
        "total_tokens": 74,
    }

    def fake_run(*args, **kwargs):
        if hasattr(mock_llm, "usage_hook") and callable(mock_llm.usage_hook):
            mock_llm.usage_hook(mock_llm.usage)
        return ""

    mock_llm.run = fake_run
    agent.llm = mock_llm

    with patch("swarms.agents.llm_manager.logger.warning") as mock_warn:
        result = agent.run("Reply with pong")
        assert result == ""
        assert mock_warn.called, "Expected logger.warning to be called when output is empty and finish_reason is length"
        warning_msg = mock_warn.call_args[0][0]
        assert "64" in warning_msg
        assert "reasoning" in warning_msg.lower()
        assert "cut off" in warning_msg.lower() or "empty" in warning_msg.lower()


def test_agent_run_warns_when_result_is_none():
    agent = Agent(
        agent_name="NoneAgent",
        model_name="openai/gpt-4o",
        max_loops=1,
        max_tokens=64,
        print_on=False,
        persistent_memory=False,
    )

    mock_llm = MagicMock()
    mock_llm.last_finish_reason = "length"
    mock_llm.last_response_model = "gpt-4o"
    mock_llm.last_response = None
    mock_llm.stream = False
    mock_llm.usage = {
        "input_tokens": 10,
        "output_tokens": 64,
        "cached_tokens": 0,
        "reasoning_tokens": 42,
        "total_tokens": 74,
    }

    def fake_run(*args, **kwargs):
        if hasattr(mock_llm, "usage_hook") and callable(mock_llm.usage_hook):
            mock_llm.usage_hook(mock_llm.usage)
        return None

    mock_llm.run = fake_run
    agent.llm = mock_llm

    with patch("swarms.agents.llm_manager.logger.warning") as mock_warn:
        agent.run("test task")
        assert mock_warn.called
        warning_msg = mock_warn.call_args[0][0]
        assert "42 reasoning tokens" in warning_msg
        assert "max_tokens=64" in warning_msg


def test_agent_call_llm_warns_on_length_cutoff_empty_output():
    agent = Agent(
        agent_name="DirectCallAgent",
        model_name="openai/gpt-4o",
        max_loops=1,
        max_tokens=128,
        print_on=False,
        persistent_memory=False,
    )

    mock_llm = MagicMock()
    mock_llm.last_finish_reason = "length"
    mock_llm.last_response_model = "gpt-4o"
    mock_llm.last_response = None
    mock_llm.stream = False
    mock_llm.usage = {
        "input_tokens": 5,
        "output_tokens": 128,
        "cached_tokens": 0,
        "reasoning_tokens": 128,
        "total_tokens": 133,
    }

    def fake_run(*args, **kwargs):
        if hasattr(mock_llm, "usage_hook") and callable(mock_llm.usage_hook):
            mock_llm.usage_hook(mock_llm.usage)
        return "   "

    mock_llm.run = fake_run
    agent.llm = mock_llm

    with patch("swarms.agents.llm_manager.logger.warning") as mock_warn:
        resp = agent.call_llm("test prompt")
        assert resp == "   "
        assert mock_warn.called
        warning_msg = mock_warn.call_args[0][0]
        assert "128 reasoning tokens" in warning_msg
        assert "max_tokens=128" in warning_msg


def test_agent_run_does_not_warn_when_output_not_empty():
    agent = Agent(
        agent_name="TestAgent",
        model_name="openai/gpt-4o",
        max_loops=1,
        max_tokens=64,
        print_on=False,
        persistent_memory=False,
    )

    mock_llm = MagicMock()
    mock_llm.last_finish_reason = "length"
    mock_llm.last_response_model = "gpt-4o"
    mock_llm.last_response = None
    mock_llm.stream = False
    mock_llm.run = MagicMock(return_value="Partial response here")
    agent.llm = mock_llm

    with patch("swarms.agents.llm_manager.logger.warning") as mock_warn:
        result = agent.run("Reply with pong")
        assert "Partial" in result
        for call in mock_warn.call_args_list:
            msg = call[0][0]
            assert "empty output" not in msg.lower()


def test_agent_run_does_not_warn_when_finish_reason_stop():
    agent = Agent(
        agent_name="TestAgent",
        model_name="openai/gpt-4o",
        max_loops=1,
        max_tokens=64,
        print_on=False,
        persistent_memory=False,
    )

    mock_llm = MagicMock()
    mock_llm.last_finish_reason = "stop"
    mock_llm.last_response_model = "gpt-4o"
    mock_llm.last_response = None
    mock_llm.stream = False
    mock_llm.run = MagicMock(return_value="pong")
    agent.llm = mock_llm

    with patch("swarms.agents.llm_manager.logger.warning") as mock_warn:
        result = agent.run("Reply with pong")
        assert "pong" in result
        mock_warn.assert_not_called()
