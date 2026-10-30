from unittest.mock import patch, MagicMock
import os
import pytest
from swarms.structs.agent import Agent


def test_provider_prefixed_model_fallback_max_tokens(monkeypatch):
    """Verify provider-prefixed models fall back to stripped model name for max_tokens."""
    def fake_get_max_tokens(name):
        if name == "llama-3.3-70b-versatile":
            return 32768
        raise Exception(f"Model {name} not mapped")

    monkeypatch.setattr("swarms.utils.llm_backend.get_max_tokens", fake_get_max_tokens)
    monkeypatch.setattr("swarms.utils.llm_backend.model_list", ["llama-3.3-70b-versatile"])
    monkeypatch.setenv("SWARMS_STRICT_MODEL_CHECK", "true")

    with patch("swarms.structs.agent.logger.warning") as mock_warn:
        agent = Agent(
            agent_name="GroqAgent",
            model_name="groq/llama-3.3-70b-versatile",
            max_loops=1,
            max_tokens=None,
            print_on=False,
            persistent_memory=False,
        )
        assert agent.max_tokens == 32768
        for call in mock_warn.call_args_list:
            msg = call[0][0]
            assert "may not be supported" not in msg
            assert "Falling back to default value of 8192" not in msg


def test_pyproject_routehub_version():
    """Verify pyproject.toml requires routehub>=0.4.0."""
    pyproject_path = os.path.join(os.path.dirname(__file__), "..", "..", "pyproject.toml")
    with open(pyproject_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert 'routehub = { version = ">=0.4.0"' in content
