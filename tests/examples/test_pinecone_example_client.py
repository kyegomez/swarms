"""Run the Pinecone example through the real SDK with offline HTTP."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("pinecone", minversion="10.0.0")
EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples/single_agent/capabilities/rag/pinecone_example.py"
)
INDEX_NAME = "financial-agent-memory"
VECTOR = [0.0] * 383 + [1.0]


@pytest.fixture
def example(monkeypatch: pytest.MonkeyPatch):
    requests = []
    existing = []
    clients = []
    index = {
        "name": INDEX_NAME,
        "host": "offline-index.pinecone.io",
        "deletion_protection": "disabled",
        "status": {"ready": True, "state": "Ready"},
        "schema": {
            "fields": {
                "_values": {
                    "type": "dense_vector",
                    "dimension": 384,
                    "metric": "cosine",
                }
            }
        },
        "deployment": {
            "deployment_type": "managed",
            "cloud": "aws",
            "region": "us-east-1",
        },
    }

    def respond(_transport, request):
        body = json.loads(request.content) if request.content else {}
        requests.append((request.method, request.url.path, body))
        if request.method == "GET" and request.url.path == "/indexes":
            response = {"indexes": existing.copy()}
        elif (request.method, request.url.path) in {
            ("GET", f"/indexes/{INDEX_NAME}"),
            ("POST", "/indexes"),
        }:
            response = index
        elif request.url.path == "/vectors/upsert":
            response = {"upsertedCount": len(body["vectors"])}
        elif request.url.path == "/query":
            response = {"matches": [{"id": "saved", "score": 1.0}]}
        else:
            pytest.fail(f"Unexpected offline request: {request.url}")
        return httpx.Response(200, json=response)

    monkeypatch.setattr(
        httpx.HTTPTransport, "handle_request", respond
    )
    blocked = Mock(side_effect=AssertionError("No network calls"))
    monkeypatch.setattr("socket.socket", blocked)
    monkeypatch.setattr("socket.create_connection", blocked)
    monkeypatch.setattr(
        "os.getenv",
        lambda name, default=None: (
            "offline-test-key"
            if name == "PINECONE_API_KEY"
            else default
        ),
    )
    monkeypatch.setitem(
        sys.modules, "dotenv", SimpleNamespace(load_dotenv=Mock())
    )
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(
            SentenceTransformer=lambda _name: SimpleNamespace(
                get_sentence_embedding_dimension=lambda: 384,
                encode=lambda _text: SimpleNamespace(
                    tolist=lambda: VECTOR.copy()
                ),
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "swarms.structs.agent",
        SimpleNamespace(
            Agent=lambda **kwargs: SimpleNamespace(
                agent_name=kwargs["agent_name"],
                run=lambda _task: "Offline answer",
            )
        ),
    )

    def load(index_exists=False):
        if index_exists:
            existing.append(index)
        result = runpy.run_path(str(EXAMPLE))
        clients.extend([result["pc"], result["pinecone_index"]])
        return result

    yield SimpleNamespace(load=load, requests=requests)
    for client in clients:
        client.close()


def test_create_index_uses_embedding_dimension(example):
    example.load()
    created = [
        body
        for method, path, body in example.requests
        if method == "POST" and path == "/indexes"
    ]
    assert len(created) == 1
    fields = created[0]["schema"]["fields"]
    assert fields["_values"]["dimension"] == len(VECTOR)
    assert fields["_values"]["metric"] == "cosine"


def test_existing_index_is_reused(example):
    example.load(index_exists=True)
    assert not any(
        method == "POST" and path == "/indexes"
        for method, path, _body in example.requests
    )


def test_agent_output_can_be_stored_and_queried(example):
    module = example.load()
    assert module["run_agent"]("Offline question") == "Offline answer"
    result = module["query_memory"]("Offline question", top_k=2)
    assert result.matches[0].id == "saved"
    upsert = next(
        body
        for _method, path, body in example.requests
        if path == "/vectors/upsert"
    )
    assert upsert["vectors"][0]["values"] == VECTOR
    assert (
        "Offline answer" in upsert["vectors"][0]["metadata"]["text"]
    )
    query = next(
        body
        for _method, path, body in example.requests
        if path == "/query"
    )
    assert query["vector"] == VECTOR
    assert query["topK"] == 2
    assert query["includeMetadata"] is True
