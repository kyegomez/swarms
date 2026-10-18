"""Run the JSON persistence example using the current Conversation API."""

import json
import runpy
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_complete_save_load_example(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(
        sys.modules, "loguru", SimpleNamespace(logger=Mock())
    )
    monkeypatch.setitem(
        sys.modules,
        "swarms.utils.generate_id",
        SimpleNamespace(generate_id=lambda: uuid.uuid4().hex),
    )
    monkeypatch.setitem(
        sys.modules,
        "swarms.utils.litellm_tokenizer",
        SimpleNamespace(
            count_tokens=lambda text, *args: len(text.split())
        ),
    )
    module = runpy.run_path(
        str(ROOT / "swarms/structs/conversation.py")
    )
    monkeypatch.setitem(
        sys.modules,
        "swarms.structs.conversation",
        SimpleNamespace(**module),
    )
    blocked = Mock(side_effect=AssertionError("No network"))
    monkeypatch.setattr("socket.socket", blocked)
    monkeypatch.setattr("socket.create_connection", blocked)
    runpy.run_path(
        str(ROOT / "examples/utils/misc/test_load_conversation.py")
    )

    saved_dir = tmp_path / "conversations"
    paths = list(saved_dir.glob("*.json"))
    assert len(paths) == 2
    saved = {
        path.stem: json.loads(path.read_text()) for path in paths
    }
    assert set(saved) == {"conversation_autosave_chat", "custom_chat"}
    assert len(saved["conversation_autosave_chat"]) == 2
    assert len(saved["custom_chat"]) == 2
    output = capsys.readouterr().out
    assert "This conversation won't be saved!" in output
    loaded_output = output.split("Loaded custom conversation:", 1)[1]
    assert "This is a custom saved conversation" in loaded_output
    assert "I'll be saved in a custom location!" in loaded_output
    assert "- conversation_autosave_chat" in loaded_output
    assert "- custom_chat" in loaded_output
