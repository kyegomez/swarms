from unittest.mock import MagicMock
import pytest

from swarms.agents.llm_manager import LLMManager


class DummyChunk:
    def __init__(self, content: str):
        delta = MagicMock()
        delta.content = content
        delta.reasoning_content = None
        delta.tool_calls = None
        choice = MagicMock()
        choice.delta = delta
        choice.finish_reason = None
        self.choices = [choice]
        self.id = "chunk-1"
        self.created = 12345
        self.model = "test-model"
        self.object = "chat.completion.chunk"
        self.system_fingerprint = "fp_1"
        self.citations = None
        self.provider_specific_fields = None
        self.service_tier = "default"
        self.obfuscation = None
        self.usage = None


class DummyLLM:
    def __init__(self, stream: bool = False):
        self.stream = stream

    def run(self, *args, **kwargs):
        if self.stream:
            def gen():
                yield DummyChunk("chunk1")
                raise ConnectionResetError("connection dropped mid-stream")
            return gen()
        return "normal text response"


class DummyAgent:
    def __init__(self):
        self.llm = DummyLLM(stream=False)
        self.agent_name = "test-agent"
        self.model_name = "test-model"
        self.print_on = False
        self.detailed_streaming = False
        self.auto_generate_prompt = False
        self.stream = False
        self.streaming_on = False


def test_run_stream_restores_stream_on_llm_run_exception():
    """_run_stream must restore agent.llm.stream if llm.run raises."""
    agent = DummyAgent()
    agent.llm.stream = False
    manager = LLMManager(agent=agent)

    def failing_run(*args, **kwargs):
        raise RuntimeError("LLM run failed")

    agent.llm.run = failing_run

    with pytest.raises(RuntimeError, match="LLM run failed"):
        manager._run_stream(task="test task", img=None)

    assert agent.llm.stream is False, f"Expected llm.stream to be False, got {agent.llm.stream}"


def test_call_detailed_streaming_restores_stream_on_mid_stream_exception():
    """_call_detailed_streaming must restore agent.llm.stream if consuming the stream fails."""
    agent = DummyAgent()
    agent.llm.stream = False
    manager = LLMManager(agent=agent)

    with pytest.raises(ConnectionResetError):
        manager._call_detailed_streaming(
            task="test", img=None, streaming_callback=None
        )

    assert agent.llm.stream is False, f"Expected llm.stream to be False, got {agent.llm.stream}"


def test_call_detailed_streaming_restores_stream_on_callback_exception():
    """_call_detailed_streaming must restore agent.llm.stream if streaming_callback raises."""
    agent = DummyAgent()
    agent.llm.stream = False
    manager = LLMManager(agent=agent)

    def gen():
        yield DummyChunk("part 1")
        yield DummyChunk("part 2")

    agent.llm.run = lambda *args, **kwargs: gen()

    def bad_callback(t):
        raise ValueError("callback crashed")

    with pytest.raises(ValueError, match="callback crashed"):
        manager._call_detailed_streaming(
            task="test", img=None, streaming_callback=bad_callback
        )

    assert agent.llm.stream is False, f"Expected llm.stream to be False, got {agent.llm.stream}"


def test_call_panel_streaming_restores_stream_on_mid_stream_exception():
    """_call_panel_streaming must restore agent.llm.stream if consuming the stream fails."""
    agent = DummyAgent()
    agent.llm.stream = False
    manager = LLMManager(agent=agent)

    with pytest.raises(ConnectionResetError):
        manager._call_panel_streaming(
            task="test", img=None, current_loop=1, streaming_callback=None
        )

    assert agent.llm.stream is False, f"Expected llm.stream to be False, got {agent.llm.stream}"


def test_call_panel_streaming_restores_stream_on_callback_exception():
    """_call_panel_streaming must restore agent.llm.stream if callback raises."""
    agent = DummyAgent()
    agent.llm.stream = False
    manager = LLMManager(agent=agent)

    def gen():
        yield DummyChunk("part 1")
        yield DummyChunk("part 2")

    agent.llm.run = lambda *args, **kwargs: gen()

    def bad_callback(t):
        raise ValueError("callback crashed")

    with pytest.raises(ValueError, match="callback crashed"):
        manager._call_panel_streaming(
            task="test", img=None, current_loop=1, streaming_callback=bad_callback
        )

    assert agent.llm.stream is False, f"Expected llm.stream to be False, got {agent.llm.stream}"


def test_subsequent_call_returns_text_not_generator_after_failed_stream():
    """Verify that after a failed streaming call, a normal call returns text, not a generator."""
    agent = DummyAgent()
    agent.llm.stream = False
    manager = LLMManager(agent=agent)

    call_count = 0

    def mock_run(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if agent.llm.stream:
            def gen():
                yield DummyChunk("chunk1")
                raise ConnectionError("network failed mid stream")
            return gen()
        return "response 2 text"

    agent.llm.run = mock_run

    # First run fails during streaming
    with pytest.raises(ConnectionError):
        manager._call_panel_streaming(
            task="first task", img=None, current_loop=1, streaming_callback=None
        )

    assert agent.llm.stream is False

    # Second run is non-streaming and succeeds
    res2 = manager.call(task="second task", img=None, current_loop=1)
    assert res2 == "response 2 text"
    assert not hasattr(res2, "__iter__") or isinstance(res2, str)
    assert agent.llm.stream is False
