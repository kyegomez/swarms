"""Bound nested lock acquisition while exercising real JSON saves."""

import json
import runpy
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples/guides/demos/synthetic_data/profession_sim"
    / "profession_persona_generator.py"
)


class BoundedLock:
    def __init__(self, lock: object) -> None:
        self.lock = lock

    def __enter__(self) -> None:
        if not self.lock.acquire(timeout=1):
            raise TimeoutError(
                "Progress lock could not be reacquired"
            )

    def __exit__(self, *args: object) -> None:
        self.lock.release()


@pytest.fixture
def generator(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    for name, module in {
        "pandas": SimpleNamespace(DataFrame=object),
        "swarms": SimpleNamespace(Agent=object),
        "loguru": SimpleNamespace(logger=Mock()),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    blocked = Mock(side_effect=AssertionError("No network"))
    monkeypatch.setattr("socket.socket", blocked)
    monkeypatch.setattr("socket.create_connection", blocked)
    cls = runpy.run_path(str(EXAMPLE))["ProfessionPersonaGenerator"]
    monkeypatch.setattr(
        cls, "_setup_signal_handlers", lambda self: None
    )
    source = tmp_path / "professions.csv"
    source.write_text("Unused offline fixture", encoding="utf-8")
    instance = cls(str(source), str(tmp_path / "personas.csv"))
    instance.progress_lock = BoundedLock(instance.progress_lock)
    return instance


def test_nested_update_saves_progress(generator: Any) -> None:
    entry = {"profession_name": "Teacher", "persona_prompt": "Teach"}
    generator._update_progress_safely(entry)
    saved = json.loads(generator.json_progress_file.read_text())
    assert saved["professions"] == [entry]
    assert saved["metadata"]["total_processed"] == 1


def test_concurrent_updates_preserve_every_entry(
    generator: Any,
) -> None:
    entries = [
        {"profession_name": str(index), "persona_prompt": "Offline"}
        for index in range(4)
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(generator._update_progress_safely, entries))
    saved = json.loads(generator.json_progress_file.read_text())
    assert (
        sorted(
            saved["professions"], key=lambda x: x["profession_name"]
        )
        == entries
    )
    assert saved["metadata"]["total_processed"] == 4


def test_direct_checkpoint_still_saves(generator: Any) -> None:
    generator._save_progress_json([], force_save=True)
    saved = json.loads(generator.json_progress_file.read_text())
    assert saved["professions"] == []
    assert saved["metadata"]["total_processed"] == 0
