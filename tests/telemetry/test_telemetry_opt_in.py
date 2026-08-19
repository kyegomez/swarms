"""Tests for telemetry opt-in gating and secret redaction.

Covers the security fix: outbound telemetry is OFF by default and
credential-bearing configuration values are redacted before serialization.
"""

import pytest

from swarms.telemetry.otel import (
    REDACTED,
    _is_secret_key,
    _sanitize,
    init_config,
    telemetry_on,
)


class _Configurable:
    """Minimal stand-in for a component whose __init__ params are captured."""

    def __init__(
        self,
        llm_api_key: str = None,
        mcp_api_key: str = None,
        model_name: str = "gpt-5.4",
        temperature: float = 0.5,
    ):
        self.llm_api_key = llm_api_key
        self.mcp_api_key = mcp_api_key
        self.model_name = model_name
        self.temperature = temperature


class TestTelemetryOptIn:
    def test_off_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("SWARMS_TELEMETRY_ON", raising=False)
        assert telemetry_on() is False

    def test_on_when_explicitly_enabled(self, monkeypatch):
        for value in ("true", "1", "yes", "on", "enable", "enabled", "TRUE"):
            monkeypatch.setenv("SWARMS_TELEMETRY_ON", value)
            assert telemetry_on() is True, value

    def test_off_for_off_values(self, monkeypatch):
        for value in ("false", "0", "no", "off", "disable", "disabled"):
            monkeypatch.setenv("SWARMS_TELEMETRY_ON", value)
            assert telemetry_on() is False, value

    def test_off_for_empty_value(self, monkeypatch):
        monkeypatch.setenv("SWARMS_TELEMETRY_ON", "")
        assert telemetry_on() is False

    def test_off_for_whitespace_value(self, monkeypatch):
        monkeypatch.setenv("SWARMS_TELEMETRY_ON", "   ")
        assert telemetry_on() is False


class TestSecretRedaction:
    def test_secret_key_detection(self):
        for name in (
            "llm_api_key",
            "mcp_api_key",
            "api_key",
            "auth_key",
            "token",
            "access_token",
            "secret",
            "password",
            "authorization",
        ):
            assert _is_secret_key(name), name
        for name in ("model_name", "temperature", "max_loops", "agent_name"):
            assert not _is_secret_key(name), name

    def test_sanitize_redacts_secret_dict_keys(self):
        payload = {
            "llm_api_key": "sk-live-secret-123",
            "model_name": "gpt-5.4",
            "nested": {"mcp_api_key": "mcp-secret", "ok": 1},
        }
        cleaned = _sanitize(payload)
        assert cleaned["llm_api_key"] == REDACTED
        assert cleaned["nested"]["mcp_api_key"] == REDACTED
        assert cleaned["model_name"] == "gpt-5.4"
        assert cleaned["nested"]["ok"] == 1

    def test_init_config_redacts_secret_params(self):
        obj = _Configurable(
            llm_api_key="sk-live-secret-123",
            mcp_api_key="mcp-secret-456",
        )
        rendered = init_config(obj)
        assert "sk-live-secret-123" not in rendered
        assert "mcp-secret-456" not in rendered
        assert REDACTED in rendered
        assert '"model_name": "gpt-5.4"' in rendered

    def test_agent_to_dict_redacts_keys(self):
        from swarms.structs.agent import Agent

        agent = Agent(
            agent_name="telemetry-redact-test",
            llm_api_key="sk-live-secret-123",
            mcp_api_key="mcp-secret-456",
            persistent_memory=False,
        )
        dumped = agent.to_dict()
        assert dumped["llm_api_key"] == "***REDACTED***"
        assert dumped["mcp_api_key"] == "***REDACTED***"
        assert "sk-live-secret-123" not in str(dumped)
        assert "mcp-secret-456" not in str(dumped)