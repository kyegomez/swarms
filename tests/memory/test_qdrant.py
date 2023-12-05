import pytest
from unittest.mock import Mock, patch

from swarms.memory.qdrant import Qdrant


@pytest.fixture
def mock_qdrant_client():
    with patch("your_module.QdrantClient") as MockQdrantClient:
        yield MockQdrantClient()


@pytest.fixture
def mock_sentence_transformer():
    with patch(
        "sentence_transformers.SentenceTransformer"
    ) as MockSentenceTransformer:
        yield MockSentenceTransformer()


@pytest.fixture
def qdrant_client(mock_qdrant_client, mock_sentence_transformer):
    client = Qdrant(api_key="your_api_key", host="your_host")
    yield client


def test_qdrant_init(qdrant_client, mock_qdrant_client):
    assert qdrant_client.client is not None


def test_load_embedding_model(
    qdrant_client, mock_sentence_transformer
):
    qdrant_client._load_embedding_model("model_name")
    mock_sentence_transformer.assert_called_once_with("model_name")


def test_setup_collection(qdrant_client, mock_qdrant_client):
    qdrant_client._setup_collection()
    mock_qdrant_client.get_collection.assert_called_once_with(
        qdrant_client.collection_name
    )


def test_add_vectors(qdrant_client, mock_qdrant_client):
    mock_doc = Mock(page_content="Sample text")
    qdrant_client.add_vectors([mock_doc])
    mock_qdrant_client.upsert.assert_called_once()


def test_search_vectors(qdrant_client, mock_qdrant_client):
    qdrant_client.search_vectors("test query")
    mock_qdrant_client.search.assert_called_once()
