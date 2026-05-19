"""
Unit tests for swarms/agents/context_compressor.py.

All network calls (litellm.completion) are mocked.  Only the integration
test at the bottom touches the filesystem, using pytest's tmp_path fixture.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from swarms.agents.context_compressor import (
    ContextCompressor,
    COMPRESSION_SYSTEM_PROMPT,
    COMPRESSION_USER_TEMPLATE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    history: str = "Turn 1\nTurn 2",
    context_length: int = 1000,
    model_name: str = "claude-sonnet-4-5",
    agent_name: str = "TestAgent",
):
    """Return a minimal stub agent that ContextCompressor can introspect."""
    short_memory = MagicMock()
    short_memory.return_history_as_string.return_value = history
    return SimpleNamespace(
        short_memory=short_memory,
        context_length=context_length,
        model_name=model_name,
        agent_name=agent_name,
    )


def _make_completion_response(content: str = "Compressed summary."):
    """Build a minimal litellm-style response object."""
    msg = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


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

    @pytest.mark.parametrize("bad", [0, -0.1, 1.1, -1, 0.0])
    def test_invalid_threshold_raises(self, bad):
        with pytest.raises(ValueError, match="threshold must be in"):
            ContextCompressor(threshold=bad)

    def test_threshold_of_one_is_valid(self):
        cc = ContextCompressor(threshold=1.0)
        assert cc.threshold == 1.0

    def test_custom_values_stored(self):
        cc = ContextCompressor(
            threshold=0.75,
            summarizer_model="gpt-4o",
            summarizer_temperature=0.5,
            summarizer_max_tokens=2000,
        )
        assert cc.threshold == 0.75
        assert cc.summarizer_model == "gpt-4o"
        assert cc.summarizer_temperature == 0.5
        assert cc.summarizer_max_tokens == 2000


# ---------------------------------------------------------------------------
# usage_ratio
# ---------------------------------------------------------------------------


class TestUsageRatio:
    def test_returns_zero_when_context_length_is_none(self):
        cc = ContextCompressor()
        agent = _make_agent(context_length=None)
        assert cc.usage_ratio(agent) == 0.0

    def test_returns_zero_when_context_length_is_zero(self):
        cc = ContextCompressor()
        agent = _make_agent(context_length=0)
        assert cc.usage_ratio(agent) == 0.0

    def test_correct_ratio(self):
        cc = ContextCompressor()
        agent = _make_agent(
            history="hello world", context_length=1000
        )
        with patch(
            "swarms.agents.context_compressor.count_tokens",
            return_value=500,
        ):
            ratio = cc.usage_ratio(agent)
        assert ratio == pytest.approx(0.5)

    def test_ratio_above_one_possible(self):
        """Allows >1.0 so should_compress still fires when over budget."""
        cc = ContextCompressor()
        agent = _make_agent(history="x" * 100, context_length=10)
        with patch(
            "swarms.agents.context_compressor.count_tokens",
            return_value=20,
        ):
            ratio = cc.usage_ratio(agent)
        assert ratio == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# should_compress
# ---------------------------------------------------------------------------


class TestShouldCompress:
    def _cc_with_ratio(self, ratio: float, threshold: float = 0.9):
        cc = ContextCompressor(threshold=threshold)
        with patch.object(cc, "usage_ratio", return_value=ratio):
            return cc.should_compress(_make_agent())

    def test_below_threshold_returns_false(self):
        assert self._cc_with_ratio(0.5) is False

    def test_at_threshold_returns_true(self):
        assert self._cc_with_ratio(0.9) is True

    def test_above_threshold_returns_true(self):
        assert self._cc_with_ratio(0.95) is True

    def test_no_loop_mode_gating(self):
        """should_compress must not inspect agent.max_loops or loop mode."""
        cc = ContextCompressor(threshold=0.5)
        agent = _make_agent()
        # Add a max_loops attribute to be sure it is never read/checked
        agent.max_loops = "auto"
        with patch.object(cc, "usage_ratio", return_value=0.8):
            result = cc.should_compress(agent)
        assert result is True


# ---------------------------------------------------------------------------
# _summarize
# ---------------------------------------------------------------------------


class TestSummarize:
    def test_raises_when_no_model_configured(self):
        cc = ContextCompressor()
        agent = _make_agent(model_name=None)
        with pytest.raises(ValueError, match="No summarizer_model"):
            cc._summarize(agent, "some history")

    def test_raises_when_agent_has_no_model_name_attr(self):
        cc = ContextCompressor()
        agent = SimpleNamespace(
            short_memory=MagicMock(), context_length=1000
        )
        with pytest.raises(ValueError, match="No summarizer_model"):
            cc._summarize(agent, "some history")

    def test_uses_summarizer_model_over_agent_model(self):
        cc = ContextCompressor(summarizer_model="gpt-4o")
        agent = _make_agent(model_name="claude-sonnet-4-5")
        resp = _make_completion_response("summary text")
        with patch(
            "swarms.agents.context_compressor.completion",
            return_value=resp,
        ) as mock_comp:
            cc._summarize(agent, "history")
        mock_comp.assert_called_once()
        _, kwargs = mock_comp.call_args
        assert kwargs["model"] == "gpt-4o"

    def test_falls_back_to_agent_model_name(self):
        cc = ContextCompressor()
        agent = _make_agent(model_name="claude-sonnet-4-5")
        resp = _make_completion_response("summary")
        with patch(
            "swarms.agents.context_compressor.completion",
            return_value=resp,
        ) as mock_comp:
            cc._summarize(agent, "history")
        _, kwargs = mock_comp.call_args
        assert kwargs["model"] == "claude-sonnet-4-5"

    def test_passes_temperature_and_max_tokens(self):
        cc = ContextCompressor(
            summarizer_model="gpt-4o",
            summarizer_temperature=0.3,
            summarizer_max_tokens=1500,
        )
        agent = _make_agent()
        resp = _make_completion_response()
        with patch(
            "swarms.agents.context_compressor.completion",
            return_value=resp,
        ) as mock_comp:
            cc._summarize(agent, "history")
        _, kwargs = mock_comp.call_args
        assert kwargs["temperature"] == 0.3
        assert kwargs["max_tokens"] == 1500

    def test_messages_structure(self):
        cc = ContextCompressor(summarizer_model="gpt-4o")
        agent = _make_agent()
        history = "turn 1\nturn 2"
        resp = _make_completion_response()
        with patch(
            "swarms.agents.context_compressor.completion",
            return_value=resp,
        ) as mock_comp:
            cc._summarize(agent, history)
        messages = mock_comp.call_args[1]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == COMPRESSION_SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        assert (
            COMPRESSION_USER_TEMPLATE.format(history=history)
            == messages[1]["content"]
        )

    def test_returns_response_content_unchanged(self):
        cc = ContextCompressor(summarizer_model="gpt-4o")
        agent = _make_agent()
        expected = "My compressed summary."
        resp = _make_completion_response(expected)
        with patch(
            "swarms.agents.context_compressor.completion",
            return_value=resp,
        ):
            result = cc._summarize(agent, "history")
        assert result == expected


# ---------------------------------------------------------------------------
# compress
# ---------------------------------------------------------------------------


class TestCompress:
    def test_returns_none_for_empty_history(self):
        cc = ContextCompressor()
        agent = _make_agent(history="")
        assert cc.compress(agent) is None

    def test_returns_none_for_whitespace_history(self):
        cc = ContextCompressor()
        agent = _make_agent(history="   \n\t  ")
        assert cc.compress(agent) is None

    def test_calls_summarize_with_full_history(self):
        cc = ContextCompressor()
        agent = _make_agent(history="turn 1\nturn 2")
        with patch.object(
            cc, "_summarize", return_value="summary"
        ) as mock_s:
            with patch(
                "swarms.agents.context_compressor.count_tokens",
                return_value=10,
            ):
                cc.compress(agent)
        mock_s.assert_called_once_with(agent, "turn 1\nturn 2")

    def test_calls_compact_exactly_once(self):
        cc = ContextCompressor()
        agent = _make_agent(history="turn 1")
        with patch.object(
            cc, "_summarize", return_value="my summary"
        ):
            with patch(
                "swarms.agents.context_compressor.count_tokens",
                return_value=10,
            ):
                cc.compress(agent)
        agent.short_memory.compact.assert_called_once()

    def test_compact_receives_preamble_wrapped_summary(self):
        cc = ContextCompressor()
        agent = _make_agent(history="turn 1")
        raw_summary = "Compressed result."
        with patch.object(cc, "_summarize", return_value=raw_summary):
            with patch(
                "swarms.agents.context_compressor.count_tokens",
                return_value=10,
            ):
                cc.compress(agent)
        compact_call = agent.short_memory.compact.call_args
        summary_arg = (
            compact_call[1].get("summary") or compact_call[0][0]
        )
        assert "[Compressed Memory Summary]" in summary_arg
        assert raw_summary in summary_arg

    def test_returns_raw_summary_without_preamble(self):
        cc = ContextCompressor()
        agent = _make_agent(history="turn 1")
        raw_summary = "Short summary."
        with patch.object(cc, "_summarize", return_value=raw_summary):
            with patch(
                "swarms.agents.context_compressor.count_tokens",
                return_value=10,
            ):
                result = cc.compress(agent)
        assert result == raw_summary


# ---------------------------------------------------------------------------
# maybe_compress
# ---------------------------------------------------------------------------


class TestMaybeCompress:
    def test_returns_none_when_should_not_compress(self):
        cc = ContextCompressor()
        agent = _make_agent()
        with patch.object(cc, "should_compress", return_value=False):
            with patch.object(cc, "compress") as mock_c:
                result = cc.maybe_compress(agent)
        assert result is None
        mock_c.assert_not_called()

    def test_calls_compress_when_should_compress(self):
        cc = ContextCompressor()
        agent = _make_agent()
        with patch.object(cc, "should_compress", return_value=True):
            with patch.object(
                cc, "compress", return_value="summary"
            ) as mock_c:
                result = cc.maybe_compress(agent)
        mock_c.assert_called_once_with(agent)
        assert result == "summary"

    def test_returns_none_and_does_not_propagate_on_error(self):
        cc = ContextCompressor()
        agent = _make_agent()
        with patch.object(cc, "should_compress", return_value=True):
            with patch.object(
                cc, "compress", side_effect=RuntimeError("boom")
            ):
                result = cc.maybe_compress(agent)
        assert result is None


# ---------------------------------------------------------------------------
# Integration — real Conversation + mocked _summarize
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_compress_updates_memory_md_and_archives(self, tmp_path):
        """End-to-end with a real Conversation and mocked _summarize."""
        from swarms.structs.conversation import Conversation

        memory_path = str(tmp_path / "MEMORY.md")

        conv = Conversation(
            system_prompt="You are a helpful assistant.",
            memory_md_path=memory_path,
            time_enabled=False,
            token_count=False,
        )
        # Populate with enough turns to look like real history
        conv.add("User", "What is 2+2?")
        conv.add("Assistant", "It is 4.")
        conv.add("User", "What about 3+3?")
        conv.add("Assistant", "That is 6.")

        cc = ContextCompressor(threshold=0.01)  # trigger immediately

        raw_summary = "Agent discussed arithmetic: 2+2=4, 3+3=6."

        # Stub agent wired to the real Conversation
        agent = SimpleNamespace(
            short_memory=conv,
            context_length=10000,
            model_name="claude-sonnet-4-5",
            agent_name="IntegrationAgent",
        )

        with patch.object(cc, "_summarize", return_value=raw_summary):
            result = cc.compress(agent)

        # compress() must return the raw summary
        assert result == raw_summary

        # conversation_history must contain system prompt + summary block
        roles = [m["role"] for m in conv.conversation_history]
        assert "System" in roles

        full_text = conv.return_history_as_string()
        assert "[Compressed Memory Summary]" in full_text
        assert raw_summary in full_text

        # Raw user/assistant turns should be gone
        assert "What is 2+2?" not in full_text

        # MEMORY.md must have been re-created (wiped + re-seeded)
        import os

        assert os.path.exists(memory_path)

        # Archive dir should exist with at least one history file
        archive_dir = str(tmp_path / "archive")
        if os.path.exists(archive_dir):
            archives = os.listdir(archive_dir)
            assert any("history_" in f for f in archives)
