"""Regression tests for LLMManager streaming flag restoration (issue #2991).

A failed streaming call must not leave llm.stream enabled. If an error occurs
during llm.run, during stream consumption, or in a streaming callback,
llm.stream must be restored to its original state so that subsequent calls
do not unexpectedly yield generators.
"""

from unittest.mock import Mock
import pytest

from swarms.agents.llm_manager import LLMManager


class DummyChunk:
    """Minimal chunk mimicking streaming response."""

    def __init__(self, text: str):
        self.choices = [
            Mock(
                delta=Mock(
                    content=text,
                    reasoning_content=None,
                    tool_calls=None,
                ),
                finish_reason=None,
                logprobs=None,
            )
        ]
        self.model = "test-model"
        self.id = "chunk-1"
        self.created = 1234567890
        self.object = "chat.completion.chunk"
        self.system_fingerprint = ""
        self.citations = None
        self.provider_specific_fields = None
        self.service_tier = "default"
        self.obfuscation = None
        self.usage = None


class FakeLLM:
    """Mock LLM tracking the stream attribute."""

    def __init__(self):
        self.stream = False
        self.model_name = "test-model"
        self.call_count = 0

    def run(self, task=None, img=None, **kwargs):
        self.call_count += 1
        return "response"


class FakeAgent:
    """Mock Agent container for LLMManager."""

    def __init__(self, llm=None):
        self.llm = llm or FakeLLM()
        self.model_name = "test-model"
        self.stream = False
        self.streaming_on = False
        self.print_on = False
        self.output_type = "final"
        self._run_streaming_callback = None
        self.streaming_callback = None
        self.llm_manager = LLMManager(self)


def test_run_stream_exception_in_llm_run_restores_stream():
    """If llm.run raises during _run_stream, llm.stream must be restored."""
    agent = FakeAgent()
    agent.llm.stream = False
    agent.llm.run = Mock(side_effect=RuntimeError("Provider connection error"))

    with pytest.raises(RuntimeError, match="Provider connection error"):
        agent.llm_manager._run_stream("test task", None)

    assert agent.llm.stream is False


def test_call_panel_streaming_exception_in_run_restores_stream():
    """If llm.run raises in _call_panel_streaming, llm.stream must be restored."""
    agent = FakeAgent()
    agent.llm.stream = False
    agent.llm.run = Mock(side_effect=RuntimeError("API timeout"))

    with pytest.raises(RuntimeError, match="API timeout"):
        agent.llm_manager._call_panel_streaming("test task", None, 0, lambda t: None)

    assert agent.llm.stream is False


def test_call_panel_streaming_exception_in_stream_restores_stream():
    """If the stream iterator raises mid-stream, llm.stream must be restored."""
    agent = FakeAgent()
    agent.llm.stream = False

    def broken_stream():
        yield "token1"
        raise ConnectionResetError("Connection dropped mid-stream")

    agent.llm.run = Mock(return_value=broken_stream())

    with pytest.raises(ConnectionResetError, match="Connection dropped mid-stream"):
        agent.llm_manager._call_panel_streaming("test task", None, 0, lambda t: None)

    assert agent.llm.stream is False


def test_call_panel_streaming_callback_exception_restores_stream():
    """If the streaming callback raises, llm.stream must be restored."""
    agent = FakeAgent()
    agent.llm.stream = False
    agent.llm.run = Mock(return_value=[DummyChunk("token1"), DummyChunk("token2")])

    def failing_callback(tok):
        raise ValueError("Websocket closed")

    with pytest.raises(ValueError, match="Websocket closed"):
        agent.llm_manager._call_panel_streaming("test task", None, 0, failing_callback)

    assert agent.llm.stream is False


def test_call_detailed_streaming_exception_in_run_restores_stream():
    """If llm.run raises in _call_detailed_streaming, llm.stream must be restored."""
    agent = FakeAgent()
    agent.llm.stream = False
    agent.llm.run = Mock(side_effect=RuntimeError("Detailed stream failed"))

    with pytest.raises(RuntimeError, match="Detailed stream failed"):
        agent.llm_manager._call_detailed_streaming("test task", None, lambda t: None)

    assert agent.llm.stream is False


def test_call_detailed_streaming_exception_in_stream_restores_stream():
    """If the stream iterator raises in _call_detailed_streaming, llm.stream must be restored."""
    agent = FakeAgent()
    agent.llm.stream = False

    def broken_stream():
        yield DummyChunk("chunk1")
        raise ConnectionResetError("Stream dropped")

    agent.llm.run = Mock(return_value=broken_stream())

    with pytest.raises(ConnectionResetError, match="Stream dropped"):
        agent.llm_manager._call_detailed_streaming("test task", None, lambda t: None)

    assert agent.llm.stream is False


def test_run_side_call_restores_stream_on_completion_and_failure():
    """Side call must restore subordinate llm.stream on success and failure."""
    agent = FakeAgent()
    agent.streaming_on = True

    subordinate_llm = FakeLLM()
    subordinate_llm.stream = False
    subordinate_llm.run = Mock(return_value=["part1", "part2"])

    agent.llm_manager.run_side_call(subordinate_llm, "summary task", current_loop=0)
    assert subordinate_llm.stream is False

    subordinate_llm.run = Mock(side_effect=RuntimeError("Subordinate error"))
    with pytest.raises(RuntimeError, match="Subordinate error"):
        agent.llm_manager.run_side_call(subordinate_llm, "summary task", current_loop=0)
    assert subordinate_llm.stream is False


def test_agent_run_stream_failure_does_not_corrupt_subsequent_run():
    """End-to-end: failed streaming run must not leave agent.llm.stream=True."""
    from swarms import Agent
    from swarms.schemas.agent_errors import AgentLLMError

    agent = Agent(
        agent_name="TestStreamAgent",
        model_name="gpt-4o-mini",
        max_loops=1,
        print_on=False,
        persistent_memory=False,
        output_type="final",
    )
    # Simulate first streaming call raising an error from provider
    agent.llm.run = Mock(side_effect=RuntimeError("Provider connection error"))
    with pytest.raises(AgentLLMError):
        agent.run("first task", streaming_callback=lambda t: None)

    assert agent.llm.stream is False

    # Second non-streaming call returns string normally
    agent.llm.run = Mock(return_value="Second task completed successfully")
    result = agent.run("second task")
    assert result == "Second task completed successfully"
    assert agent.llm.stream is False
