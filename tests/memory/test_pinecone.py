import os
from unittest.mock import patch
from swarms.memory.pinecone import PineconeVectorStore

api_key = os.getenv("PINECONE_API_KEY") or ""


def test_init():
    with patch("pinecone.init") as MockInit, patch(
        "pinecone.Index"
    ) as MockIndex:
        store = PineconeVectorStore(
            api_key=api_key,
            index_name="test_index",
            environment="test_env",
        )
        MockInit.assert_called_once()
        MockIndex.assert_called_once()
        assert store.index == MockIndex.return_value


def test_upsert_vector():
    with patch("pinecone.init"), patch("pinecone.Index") as MockIndex:
        store = PineconeVectorStore(
            api_key=api_key,
            index_name="test_index",
            environment="test_env",
        )
        store.upsert_vector(
            [1.0, 2.0, 3.0],
            "test_id",
            "test_namespace",
            {"meta": "data"},
        )
        MockIndex.return_value.upsert.assert_called()


def test_load_entry():
    with patch("pinecone.init"), patch("pinecone.Index") as MockIndex:
        store = PineconeVectorStore(
            api_key=api_key,
            index_name="test_index",
            environment="test_env",
        )
        store.load_entry("test_id", "test_namespace")
        MockIndex.return_value.fetch.assert_called()


def test_load_entries():
    with patch("pinecone.init"), patch("pinecone.Index") as MockIndex:
        store = PineconeVectorStore(
            api_key=api_key,
            index_name="test_index",
            environment="test_env",
        )
        store.load_entries("test_namespace")
        MockIndex.return_value.query.assert_called()


def test_query():
    with patch("pinecone.init"), patch("pinecone.Index") as MockIndex:
        store = PineconeVectorStore(
            api_key=api_key,
            index_name="test_index",
            environment="test_env",
        )
        store.query("test_query", 10, "test_namespace")
        MockIndex.return_value.query.assert_called()


def test_create_index():
    with patch("pinecone.init"), patch("pinecone.Index"), patch(
        "pinecone.create_index"
    ) as MockCreateIndex:
        store = PineconeVectorStore(
            api_key=api_key,
            index_name="test_index",
            environment="test_env",
        )
        store.create_index("test_index")
        MockCreateIndex.assert_called()
