"""Verify example vector retrieval with the real in-memory Qdrant client."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("qdrant_client")
EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples/single_agent/capabilities/rag/qdrant_agent.py"
)


@pytest.fixture
def memory(monkeypatch: pytest.MonkeyPatch):
    blocked = Mock(side_effect=AssertionError("No network calls"))
    monkeypatch.setattr("socket.socket", blocked)
    monkeypatch.setattr("socket.create_connection", blocked)
    monkeypatch.setattr("os.getenv", lambda *_args: None)
    monkeypatch.setitem(
        sys.modules, "swarms", SimpleNamespace(Agent=Mock())
    )
    example = runpy.run_path(str(EXAMPLE))
    store = example["QdrantMemory"](vector_size=3)
    try:
        yield store
    finally:
        store.client.close()
        example["memory"].client.close()


def test_query_returns_ranked_payload_and_similarity(memory):
    memory.add("Exact", [1.0, 0.0, 0.0], {"kind": "reference"})
    memory.add("Related", [0.8, 0.6, 0.0])
    memory.add("Unrelated", [0.0, 1.0, 0.0])

    results = memory.query([1.0, 0.0, 0.0])

    assert [item["text"] for item in results] == ["Exact", "Related"]
    assert results[0]["kind"] == "reference"
    assert results[0]["similarity_score"] == pytest.approx(1.0)
    assert results[1]["similarity_score"] == pytest.approx(0.8)


def test_query_honors_limit_and_score_threshold(memory):
    memory.add("Exact", [1.0, 0.0, 0.0])
    memory.add("Related", [0.8, 0.6, 0.0])

    assert len(memory.query([1.0, 0.0, 0.0], limit=1)) == 1
    assert (
        len(memory.query([1.0, 0.0, 0.0], score_threshold=0.9)) == 1
    )


def test_query_after_delete_and_clear(memory):
    memory_id = memory.add("Temporary", [1.0, 0.0, 0.0])
    memory.delete(memory_id)
    assert memory.query([1.0, 0.0, 0.0]) == []
    memory.add("Temporary again", [1.0, 0.0, 0.0])
    memory.clear()
    assert memory.query([1.0, 0.0, 0.0]) == []


def test_empty_collection_returns_no_results(memory):
    assert memory.query([1.0, 0.0, 0.0]) == []
