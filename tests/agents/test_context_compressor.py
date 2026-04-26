"""
Unit tests for swarms/agents/context_compressor.py.

All tests are network-free: litellm.completion is mocked wherever the real
LLM would be called.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from swarms.agents.context_compressor import ContextCompressor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    history: str = "turn 1\nturn 2",
    context_length: int = 1000,
    model_name: str = "gpt-4o",
    agent_name: str = "test-agent",
) -> MagicMock:
    """Return a minimal mock agent that satisfies ContextCompressor's API."""
    agent = MagicMock()
    agent.context_length = context_length
    agent.model_name = model_name
    agent.agent_name = agent_name
    agent.short_memory.return_history_as_string.return_value = history
    return agent


def _make_completion_response(content: str) -> MagicMock:
    """Build the minimal litellm response shape."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# __init__ validation
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_values(self):
        cc = ContextCompressor()
        assert cc.threshold == 0.9
        assert cc.summarizer_temperature == 0.2
        assert cc.summarizer_max_tokens == 4000
        assert cc.summarizer_model is None

    def test_threshold_zero_raises(self):
        with pytest.raises(ValueError):
            ContextCompressor(threshold=0)

    def test_threshold_negative_raises(self):
        with pytest.raises(ValueError):
            ContextCompressor(threshold=-0.1)

    def test_threshold_above_one_raises(self):
        with pytest.raises(ValueError):
            ContextCompressor(threshold=1.1)

    def test_threshold_exactly_one_is_valid(self):
        cc = ContextCompressor(threshold=1.0)
        assert cc.threshold == 1.0


# ---------------------------------------------------------------------------
# usage_ratio
# ---------------------------------------------------------------------------


class TestUsageRatio:
    def test_returns_zero_when_context_length_is_none(self):
        agent = _make_agent(context_length=None)
        cc = ContextCompressor()
        assert cc.usage_ratio(agent) == 0.0

    def test_returns_zero_when_context_length_is_zero(self):
        agent = _make_agent(context_length=0)
        cc = ContextCompressor()
        assert cc.usage_ratio(agent) == 0.0

    def test_ratio_for_populated_history(self):
        """usage_ratio = count_tokens(history) / context_length."""
        agent = _make_agent(history="hello world", context_length=1000)
        cc = ContextCompressor()

        with patch(
            "swarms.agents.context_compressor.count_tokens",
            return_value=100,
        ):
            ratio = cc.usage_ratio(agent)

        assert ratio == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# should_compress
# ---------------------------------------------------------------------------


class TestShouldCompress:
    def test_false_when_below_threshold(self):
        cc = ContextCompressor(threshold=0.9)
        with patch.object(cc, "usage_ratio", return_value=0.5):
            assert cc.should_compress(MagicMock()) is False

    def test_true_at_exactly_threshold(self):
        cc = ContextCompressor(threshold=0.9)
        with patch.object(cc, "usage_ratio", return_value=0.9):
            assert cc.should_compress(MagicMock()) is True

    def test_true_above_threshold(self):
        cc = ContextCompressor(threshold=0.9)
        with patch.object(cc, "usage_ratio", return_value=0.95):
            assert cc.should_compress(MagicMock()) is True

    def test_no_max_loops_mode_gating(self):
        """should_compress must not check agent.max_loops; it only measures
        the token budget regardless of loop mode."""
        cc = ContextCompressor(threshold=0.5)
        agent = _make_agent()
        agent.max_loops = 3  # integer loop mode — should still fire

        with patch.object(cc, "usage_ratio", return_value=0.8):
            assert cc.should_compress(agent) is True


# ---------------------------------------------------------------------------
# _summarize
# ---------------------------------------------------------------------------


class TestSummarize:
    def test_raises_when_no_model_available(self):
        cc = ContextCompressor()
        agent = MagicMock()
        agent.model_name = None

        with pytest.raises(ValueError, match="No summarizer_model"):
            cc._summarize(agent, "some history")

    def test_uses_summarizer_model_over_agent_model(self):
        cc = ContextCompressor(summarizer_model="gpt-4o-mini")
        agent = _make_agent(model_name="gpt-4o")

        fake_resp = _make_completion_response("compressed")
        with patch(
            "swarms.agents.context_compressor.completion",
            return_value=fake_resp,
        ) as mock_completion:
            result = cc._summarize(agent, "history text")

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert result == "compressed"

    def test_falls_back_to_agent_model(self):
        cc = ContextCompressor()
        agent = _make_agent(model_name="gpt-4.1")

        fake_resp = _make_completion_response("summary")
        with patch(
            "swarms.agents.context_compressor.completion",
            return_value=fake_resp,
        ) as mock_completion:
            cc._summarize(agent, "history text")

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["model"] == "gpt-4.1"

    def test_passes_temperature_and_max_tokens(self):
        cc = ContextCompressor(
            summarizer_model="gpt-4o",
            summarizer_temperature=0.5,
            summarizer_max_tokens=512,
        )
        agent = _make_agent()

        fake_resp = _make_completion_response("ok")
        with patch(
            "swarms.agents.context_compressor.completion",
            return_value=fake_resp,
        ) as mock_completion:
            cc._summarize(agent, "history")

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 512

    def test_returns_message_content_unchanged(self):
        cc = ContextCompressor(summarizer_model="gpt-4o")
        agent = _make_agent()

        expected = "  raw summary with leading spaces  "
        fake_resp = _make_completion_response(expected)
        with patch(
            "swarms.agents.context_compressor.completion",
            return_value=fake_resp,
        ):
            result = cc._summarize(agent, "history")

        assert result == expected


# ---------------------------------------------------------------------------
# compress
# ---------------------------------------------------------------------------


class TestCompress:
    def test_returns_none_when_history_empty(self):
        cc = ContextCompressor(summarizer_model="gpt-4o")
        agent = _make_agent(history="")
        assert cc.compress(agent) is None

    def test_returns_none_when_history_whitespace_only(self):
        cc = ContextCompressor(summarizer_model="gpt-4o")
        agent = _make_agent(history="   \n\t  ")
        assert cc.compress(agent) is None

    def test_calls_summarize_with_full_history(self):
        cc = ContextCompressor(summarizer_model="gpt-4o")
        agent = _make_agent(history="turn A\nturn B")

        with patch.object(
            cc, "_summarize", return_value="summary text"
        ) as mock_summarize:
            with patch(
                "swarms.agents.context_compressor.count_tokens",
                return_value=10,
            ):
                cc.compress(agent)

        mock_summarize.assert_called_once_with(agent, "turn A\nturn B")

    def test_calls_compact_exactly_once(self):
        cc = ContextCompressor(summarizer_model="gpt-4o")
        agent = _make_agent(history="some history")

        with patch.object(cc, "_summarize", return_value="summary"):
            with patch(
                "swarms.agents.context_compressor.count_tokens",
                return_value=10,
            ):
                cc.compress(agent)

        agent.short_memory.compact.assert_called_once()

    def test_compact_receives_preamble_wrapped_summary(self):
        cc = ContextCompressor(summarizer_model="gpt-4o")
        agent = _make_agent(history="some history")

        with patch.object(cc, "_summarize", return_value="the summary"):
            with patch(
                "swarms.agents.context_compressor.count_tokens",
                return_value=10,
            ):
                cc.compress(agent)

        call_args = agent.short_memory.compact.call_args
        summary_arg = call_args[1].get("summary") or call_args[0][0]
        assert "[Compressed Memory Summary]" in summary_arg
        assert "the summary" in summary_arg

    def test_returns_raw_summary_without_preamble(self):
        cc = ContextCompressor(summarizer_model="gpt-4o")
        agent = _make_agent(history="some history")

        with patch.object(cc, "_summarize", return_value="raw summary"):
            with patch(
                "swarms.agents.context_compressor.count_tokens",
                return_value=10,
            ):
                result = cc.compress(agent)

        assert result == "raw summary"


# ---------------------------------------------------------------------------
# maybe_compress
# ---------------------------------------------------------------------------


class TestMaybeCompress:
    def test_returns_none_when_should_compress_false(self):
        cc = ContextCompressor()
        agent = _make_agent()

        with patch.object(cc, "should_compress", return_value=False):
            with patch.object(cc, "compress") as mock_compress:
                result = cc.maybe_compress(agent)

        assert result is None
        mock_compress.assert_not_called()

    def test_calls_compress_when_should_compress_true(self):
        cc = ContextCompressor()
        agent = _make_agent()

        with patch.object(cc, "should_compress", return_value=True):
            with patch.object(
                cc, "compress", return_value="summary"
            ) as mock_compress:
                result = cc.maybe_compress(agent)

        mock_compress.assert_called_once_with(agent)
        assert result == "summary"

    def test_returns_none_and_logs_error_on_compress_exception(self):
        cc = ContextCompressor()
        agent = _make_agent()

        with patch.object(cc, "should_compress", return_value=True):
            with patch.object(
                cc, "compress", side_effect=RuntimeError("boom")
            ):
                result = cc.maybe_compress(agent)

        assert result is None

    def test_does_not_propagate_exception_from_compress(self):
        cc = ContextCompressor()
        agent = _make_agent()

        with patch.object(cc, "should_compress", return_value=True):
            with patch.object(
                cc, "compress", side_effect=Exception("unexpected")
            ):
                # Must not raise
                cc.maybe_compress(agent)
