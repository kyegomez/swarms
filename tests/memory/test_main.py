import pytest
from unittest.mock import Mock
from swarms.memory.ocean import OceanDB


@pytest.fixture
def mock_ocean_client():
    return Mock()


@pytest.fixture
def mock_collection():
    return Mock()


@pytest.fixture
def ocean_db(mock_ocean_client):
    OceanDB.client = mock_ocean_client
    return OceanDB()


def test_init(ocean_db, mock_ocean_client):
    mock_ocean_client.heartbeat.return_value = "OK"
    assert ocean_db.client.heartbeat() == "OK"


def test_create_collection(
    ocean_db, mock_ocean_client, mock_collection
):
    mock_ocean_client.create_collection.return_value = mock_collection
    collection = ocean_db.create_collection("test", "text")
    assert collection == mock_collection


def test_append_document(ocean_db, mock_collection):
    document = "test_document"
    id = "test_id"
    ocean_db.append_document(mock_collection, document, id)
    mock_collection.add.assert_called_once_with(
        documents=[document], ids=[id]
    )


def test_add_documents(ocean_db, mock_collection):
    documents = ["test_document1", "test_document2"]
    ids = ["test_id1", "test_id2"]
    ocean_db.add_documents(mock_collection, documents, ids)
    mock_collection.add.assert_called_once_with(
        documents=documents, ids=ids
    )


def test_query(ocean_db, mock_collection):
    query_texts = ["test_query"]
    n_results = 10
    mock_collection.query.return_value = "query_result"
    result = ocean_db.query(mock_collection, query_texts, n_results)
    assert result == "query_result"
