"""Offline regressions for the Mem0 example's text chunking."""

import os
import runpy
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest


@pytest.fixture
def chunk_text(monkeypatch: pytest.MonkeyPatch) -> Callable:
    """Load the actual example with credentials and clients stubbed."""
    modules = {
        name: ModuleType(name)
        for name in ("mem0", "swarms", "dotenv")
    }
    modules["mem0"].MemoryClient = Mock()
    modules["swarms"].Agent = Mock()
    modules["dotenv"].load_dotenv = Mock()
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(os, "getenv", Mock(return_value=None))
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/guides/mem0/mem0_and_swarms.py"
    )
    example = runpy.run_path(str(path))
    modules["mem0"].MemoryClient.assert_not_called()
    modules["swarms"].Agent.assert_not_called()
    return example["chunk_text"]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (5, -1), (5, 5), (5, 6), (200, 200)],
)
def test_invalid_configuration_rejected_even_for_empty_text(
    chunk_text: Callable, chunk_size: int, overlap: int
) -> None:
    """Reject parameters that prevent cursor progress or skip content."""
    with pytest.raises(ValueError):
        chunk_text("", chunk_size=chunk_size, overlap=overlap)


@pytest.mark.parametrize(
    ("text", "chunk_size", "overlap", "expected"),
    [
        ("", 5, 0, [""]),
        ("abc", 5, 0, ["abc"]),
        ("abcdef", 3, 0, ["abc", "def"]),
        ("abcdefgh", 4, 1, ["abcd", "defg", "gh"]),
        ("甲乙丙丁戊己", 3, 1, ["甲乙丙", "丙丁戊", "戊己"]),
    ],
)
def test_valid_chunks_preserve_content_and_overlap(
    chunk_text: Callable,
    text: str,
    chunk_size: int,
    overlap: int,
    expected: list[str],
) -> None:
    """Keep valid empty, short, nonoverlapping and Unicode chunks."""
    assert chunk_text(text, chunk_size, overlap) == expected
