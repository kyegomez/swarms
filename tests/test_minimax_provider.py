"""
Tests for MiniMax provider integration in Swarms.

Unit tests verify provider configuration and auto-routing without
making actual API calls. Integration tests validate real API
connectivity (skipped unless MINIMAX_API_KEY is set).
"""

import os
from unittest.mock import patch

import pytest

from swarms.utils.litellm_wrapper import LiteLLM
from swarms.structs.model_router import (
    model_recommendations,
    providers,
)
from swarms.cli.utils import _detect_active_provider, check_api_keys


# ---------------------------------------------------------------------------
# Unit tests – model_router.py
# ---------------------------------------------------------------------------


class TestModelRouterMiniMax:
    """Verify MiniMax is registered in model_router metadata."""

    def test_minimax_m27_in_recommendations(self):
        assert "minimax/MiniMax-M2.7" in model_recommendations

    def test_minimax_m25_highspeed_in_recommendations(self):
        assert "minimax/MiniMax-M2.5-highspeed" in model_recommendations

    def test_minimax_m27_provider_field(self):
        entry = model_recommendations["minimax/MiniMax-M2.7"]
        assert entry["provider"] == "minimax"

    def test_minimax_m25_highspeed_provider_field(self):
        entry = model_recommendations["minimax/MiniMax-M2.5-highspeed"]
        assert entry["provider"] == "minimax"

    def test_minimax_m27_has_best_for(self):
        entry = model_recommendations["minimax/MiniMax-M2.7"]
        assert isinstance(entry["best_for"], list)
        assert len(entry["best_for"]) > 0

    def test_minimax_in_providers_dict(self):
        assert "minimax" in str(providers).lower()

    def test_minimax_m27_description_nonempty(self):
        entry = model_recommendations["minimax/MiniMax-M2.7"]
        assert len(entry["description"]) > 0


# ---------------------------------------------------------------------------
# Unit tests – cli/utils.py provider detection
# ---------------------------------------------------------------------------


class TestCLIProviderDetection:
    """Verify MiniMax API key detection in CLI utilities."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}, clear=False)
    def test_detect_active_provider_includes_minimax(self):
        result = _detect_active_provider()
        assert "MiniMax" in result

    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
            "GROQ_API_KEY": "",
            "GOOGLE_API_KEY": "",
            "COHERE_API_KEY": "",
            "MISTRAL_API_KEY": "",
            "TOGETHER_API_KEY": "",
            "MINIMAX_API_KEY": "test-key",
        },
    )
    def test_minimax_only_provider(self):
        result = _detect_active_provider()
        assert "MiniMax" in result

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}, clear=False)
    def test_check_api_keys_includes_minimax(self):
        success, icon, msg = check_api_keys()
        assert success is True
        assert "MINIMAX_API_KEY" in msg


# ---------------------------------------------------------------------------
# Unit tests – LiteLLM wrapper MiniMax auto-routing
# ---------------------------------------------------------------------------


class TestLiteLLMWrapperMiniMax:
    """Verify MiniMax auto-routing in the LiteLLM wrapper."""

    def test_minimax_prefix_sets_base_url(self):
        llm = LiteLLM(model_name="minimax/MiniMax-M2.7")
        assert llm.base_url == "https://api.minimax.io/v1"

    def test_minimax_prefix_routes_via_openai(self):
        """Model name should be rewritten to openai/<model> for litellm."""
        llm = LiteLLM(model_name="minimax/MiniMax-M2.7")
        assert llm.model_name == "openai/MiniMax-M2.7"

    def test_minimax_prefix_case_insensitive(self):
        llm = LiteLLM(model_name="MiniMax/MiniMax-M2.5")
        assert llm.base_url == "https://api.minimax.io/v1"
        assert llm.model_name == "openai/MiniMax-M2.5"

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "env-test-key"}, clear=False)
    def test_minimax_prefix_reads_env_api_key(self):
        llm = LiteLLM(model_name="minimax/MiniMax-M2.7")
        assert llm.api_key == "env-test-key"

    def test_minimax_prefix_explicit_api_key_takes_precedence(self):
        llm = LiteLLM(
            model_name="minimax/MiniMax-M2.7",
            api_key="explicit-key",
        )
        assert llm.api_key == "explicit-key"

    def test_minimax_prefix_explicit_base_url_takes_precedence(self):
        llm = LiteLLM(
            model_name="minimax/MiniMax-M2.7",
            base_url="https://custom.api.com/v1",
        )
        assert llm.base_url == "https://custom.api.com/v1"

    def test_minimax_temperature_clamping_above_1(self):
        llm = LiteLLM(
            model_name="minimax/MiniMax-M2.7",
            temperature=1.5,
        )
        assert llm.temperature == 1.0

    def test_minimax_temperature_no_clamp_within_range(self):
        llm = LiteLLM(
            model_name="minimax/MiniMax-M2.7",
            temperature=0.7,
        )
        assert llm.temperature == 0.7

    def test_minimax_temperature_zero_allowed(self):
        llm = LiteLLM(
            model_name="minimax/MiniMax-M2.7",
            temperature=0.0,
        )
        assert llm.temperature == 0.0

    def test_minimax_highspeed_model_routing(self):
        llm = LiteLLM(model_name="minimax/MiniMax-M2.5-highspeed")
        assert llm.model_name == "openai/MiniMax-M2.5-highspeed"
        assert llm.base_url == "https://api.minimax.io/v1"

    def test_non_minimax_model_no_routing(self):
        llm = LiteLLM(model_name="gpt-4.1")
        assert llm.base_url is None
        assert llm.model_name == "gpt-4.1"

    def test_minimax_system_prompt_preserved(self):
        llm = LiteLLM(
            model_name="minimax/MiniMax-M2.7",
            system_prompt="You are a helpful assistant.",
        )
        assert any(
            m.get("content") == "You are a helpful assistant."
            for m in llm.messages
        )

    def test_minimax_max_tokens_preserved(self):
        llm = LiteLLM(
            model_name="minimax/MiniMax-M2.7",
            max_tokens=8192,
        )
        assert llm.max_tokens == 8192

    def test_minimax_stream_option_preserved(self):
        llm = LiteLLM(
            model_name="minimax/MiniMax-M2.7",
            stream=True,
        )
        assert llm.stream is True

    def test_minimax_default_temperature(self):
        llm = LiteLLM(model_name="minimax/MiniMax-M2.7")
        assert llm.temperature == 0.5  # default


# ---------------------------------------------------------------------------
# Integration tests – require MINIMAX_API_KEY
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.getenv("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set",
)
class TestMiniMaxIntegration:
    """Integration tests that call the real MiniMax API."""

    def test_minimax_m27_simple_completion(self):
        llm = LiteLLM(
            model_name="minimax/MiniMax-M2.7",
            max_tokens=64,
            temperature=0.1,
        )
        response = llm.run("Say hello in exactly three words.")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_minimax_m25_highspeed_completion(self):
        llm = LiteLLM(
            model_name="minimax/MiniMax-M2.5-highspeed",
            max_tokens=64,
            temperature=0.1,
        )
        response = llm.run("What is 2 + 2?")
        assert isinstance(response, str)
        assert "4" in response

    def test_minimax_with_system_prompt(self):
        llm = LiteLLM(
            model_name="minimax/MiniMax-M2.7",
            system_prompt="Always respond in JSON format with a 'result' key.",
            max_tokens=128,
            temperature=0.1,
        )
        response = llm.run("What is the capital of France?")
        assert isinstance(response, str)
        assert len(response) > 0
