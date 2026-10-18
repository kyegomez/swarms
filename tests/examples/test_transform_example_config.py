"""Run advanced scenarios through real message transforms offline."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = (
    ROOT
    / "examples/single_agent/utils/transform_prompts"
    / "transforms_examples.py"
)


@pytest.fixture
def results(monkeypatch):
    blocked = Mock(side_effect=AssertionError("No network access"))
    monkeypatch.setattr("socket.socket", blocked)
    monkeypatch.setattr("socket.create_connection", blocked)
    monkeypatch.setitem(
        sys.modules, "swarms", SimpleNamespace(Agent=Mock())
    )
    monkeypatch.setitem(
        sys.modules,
        "swarms.structs.conversation",
        SimpleNamespace(Conversation=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "swarms.utils",
        SimpleNamespace(
            llm_backend=SimpleNamespace(
                get_model_info=lambda _model: {
                    "max_input_tokens": 8192
                }
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "swarms.utils.litellm_tokenizer",
        SimpleNamespace(
            count_tokens=lambda text, _model: len(text.split())
        ),
    )
    monkeypatch.setitem(
        sys.modules, "loguru", SimpleNamespace(logger=Mock())
    )
    core = runpy.run_path(str(ROOT / "swarms/structs/transforms.py"))
    monkeypatch.setitem(
        sys.modules,
        "swarms.structs.transforms",
        SimpleNamespace(**core),
    )
    observed = []
    transform = core["MessageTransforms"].transform_messages

    def capture(instance, messages, *args, **kwargs):
        result = transform(instance, messages, *args, **kwargs)
        observed.append((instance.config, result))
        return result

    monkeypatch.setattr(
        core["MessageTransforms"], "transform_messages", capture
    )
    runpy.run_path(str(EXAMPLE))["example_5_advanced_scenarios"]()
    return observed


def test_token_heavy_scenario_applies_its_configured_limit(results):
    config, result = results[1]
    assert result.original_token_count > 2000
    assert result.compressed_token_count <= 2000
    assert result.was_compressed
    assert config.max_tokens == 2000


def test_message_count_scenario_preserves_its_limit(results):
    config, result = results[0]
    assert config.max_messages == 20
    assert result.original_message_count == 151
    assert result.compressed_message_count == 20


def test_short_mixed_content_is_preserved(results):
    _config, result = results[2]
    assert result.original_message_count == 5
    assert result.compressed_message_count == 5
    assert not result.was_compressed
