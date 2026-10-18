"""Check retrieval counts using LlamaIndex's actual local vector store."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

core = pytest.importorskip("llama_index.core")
MockEmbedding = pytest.importorskip(
    "llama_index.core.embeddings"
).MockEmbedding
MockLLM = pytest.importorskip("llama_index.core.llms").MockLLM
RetrieverQueryEngine = pytest.importorskip(
    "llama_index.core.query_engine"
).RetrieverQueryEngine
ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = [
    "examples/single_agent/capabilities/rag/full_agent_rag_example.py",
    "examples/guides/demos/spike/memory.py",
]


@pytest.fixture(params=EXAMPLES)
def example(monkeypatch, tmp_path, request):
    monkeypatch.chdir(tmp_path)
    documents = tmp_path / "docs"
    documents.mkdir()
    for i in range(6):
        (documents / f"{i}.txt").write_text(
            f"Offline reference document {i}.", encoding="utf-8"
        )
    monkeypatch.setattr(
        core.Settings, "_embed_model", MockEmbedding(embed_dim=3)
    )
    monkeypatch.setattr(core.Settings, "_llm", MockLLM())
    monkeypatch.setattr(core.Settings, "_transformations", [])
    monkeypatch.setattr(core.Settings, "_tokenizer", str.split)
    blocked = Mock(side_effect=AssertionError("No network calls"))
    monkeypatch.setattr("socket.socket.connect", blocked)
    monkeypatch.setattr("socket.create_connection", blocked)
    monkeypatch.setitem(
        sys.modules, "dotenv", SimpleNamespace(load_dotenv=Mock())
    )
    monkeypatch.setitem(
        sys.modules, "loguru", SimpleNamespace(logger=Mock())
    )
    monkeypatch.setitem(
        sys.modules,
        "swarms",
        SimpleNamespace(
            Agent=lambda **kwargs: SimpleNamespace(**kwargs),
            AgentRearrange=Mock(),
        ),
    )
    observed = []
    query = RetrieverQueryEngine.query

    def observe(engine, query_text):
        response = query(engine, query_text)
        observed.append(len(response.source_nodes))
        return response

    monkeypatch.setattr(RetrieverQueryEngine, "query", observe)
    module = runpy.run_path(str(ROOT / request.param))
    return module["LlamaIndexDB"], documents, observed


@pytest.mark.parametrize(
    ("configured", "override", "expected"),
    [(4, None, 4), (4, 1, 1), (None, None, 2)],
)
def test_query_retrieves_configured_number_of_documents(
    example, configured, override, expected
):
    database_type, documents, observed = example
    options = (
        {} if configured is None else {"similarity_top_k": configured}
    )
    database = database_type(data_dir=str(documents), **options)
    query_options = (
        {} if override is None else {"similarity_top_k": override}
    )

    result = database.query(
        "Find offline references", **query_options
    )

    assert isinstance(result, str)
    assert observed == [expected]
