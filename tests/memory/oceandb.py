import pytest
from unittest.mock import Mock, patch
from swarms.memory.oceandb import OceanDB


def test_init():
    with patch("oceandb.Client") as MockClient:
        MockClient.return_value.heartbeat.return_value = "OK"
        db = OceanDB(MockClient)
        MockClient.assert_called_once()
        assert db.client == MockClient


def test_init_exception():
    with patch("oceandb.Client") as MockClient:
        MockClient.side_effect = Exception("Client error")
        with pytest.raises(Exception) as e:
            db = OceanDB(MockClient)
        assert str(e.value) == "Client error"


def test_create_collection():
    with patch("oceandb.Client") as MockClient:
        db = OceanDB(MockClient)
        db.create_collection("test", "modality")
        MockClient.create_collection.assert_called_once_with(
            "test", embedding_function=Mock.ANY
        )


def test_create_collection_exception():
    with patch("oceandb.Client") as MockClient:
        MockClient.create_collection.side_effect = Exception("Create collection error")
        db = OceanDB(MockClient)
        with pytest.raises(Exception) as e:
            db.create_collection("test", "modality")
        assert str(e.value) == "Create collection error"


def test_append_document():
    with patch("oceandb.Client") as MockClient:
        db = OceanDB(MockClient)
        collection = Mock()
        db.append_document(collection, "doc", "id")
        collection.add.assert_called_once_with(documents=["doc"], ids=["id"])


def test_append_document_exception():
    with patch("oceandb.Client") as MockClient:
        db = OceanDB(MockClient)
        collection = Mock()
        collection.add.side_effect = Exception("Append document error")
        with pytest.raises(Exception) as e:
            db.append_document(collection, "doc", "id")
        assert str(e.value) == "Append document error"


def test_add_documents():
    with patch("oceandb.Client") as MockClient:
        db = OceanDB(MockClient)
        collection = Mock()
        db.add_documents(collection, ["doc1", "doc2"], ["id1", "id2"])
        collection.add.assert_called_once_with(
            documents=["doc1", "doc2"], ids=["id1", "id2"]
        )


def test_add_documents_exception():
    with patch("oceandb.Client") as MockClient:
        db = OceanDB(MockClient)
        collection = Mock()
        collection.add.side_effect = Exception("Add documents error")
        with pytest.raises(Exception) as e:
            db.add_documents(collection, ["doc1", "doc2"], ["id1", "id2"])
        assert str(e.value) == "Add documents error"


def test_query():
    with patch("oceandb.Client") as MockClient:
        db = OceanDB(MockClient)
        collection = Mock()
        db.query(collection, ["query1", "query2"], 2)
        collection.query.assert_called_once_with(
            query_texts=["query1", "query2"], n_results=2
        )


def test_query_exception():
    with patch("oceandb.Client") as MockClient:
        db = OceanDB(MockClient)
        collection = Mock()
        collection.query.side_effect = Exception("Query error")
        with pytest.raises(Exception) as e:
            db.query(collection, ["query1", "query2"], 2)
        assert str(e.value) == "Query error"
