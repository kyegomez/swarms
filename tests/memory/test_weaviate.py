import pytest
from unittest.mock import Mock, patch
from swarms.memory.weaviate import WeaviateClient


# Define fixture for a WeaviateClient instance with mocked methods
@pytest.fixture
def weaviate_client_mock():
    client = WeaviateClient(
        http_host="mock_host",
        http_port="mock_port",
        http_secure=False,
        grpc_host="mock_grpc_host",
        grpc_port="mock_grpc_port",
        grpc_secure=False,
        auth_client_secret="mock_api_key",
        additional_headers={
            "X-OpenAI-Api-Key": "mock_openai_api_key"
        },
        additional_config=Mock(),
    )

    # Mock the methods
    client.client.collections.create = Mock()
    client.client.collections.get = Mock()
    client.client.collections.query = Mock()
    client.client.collections.data.insert = Mock()
    client.client.collections.data.update = Mock()
    client.client.collections.data.delete_by_id = Mock()

    return client


# Define tests for the WeaviateClient class
def test_create_collection(weaviate_client_mock):
    # Test creating a collection
    weaviate_client_mock.create_collection(
        "test_collection", [{"name": "property"}]
    )
    weaviate_client_mock.client.collections.create.assert_called_with(
        name="test_collection",
        vectorizer_config=None,
        properties=[{"name": "property"}],
    )


def test_add_object(weaviate_client_mock):
    # Test adding an object
    properties = {"name": "John"}
    weaviate_client_mock.add("test_collection", properties)
    weaviate_client_mock.client.collections.get.assert_called_with(
        "test_collection"
    )
    weaviate_client_mock.client.collections.data.insert.assert_called_with(
        properties
    )


def test_query_objects(weaviate_client_mock):
    # Test querying objects
    query = "name:John"
    weaviate_client_mock.query("test_collection", query)
    weaviate_client_mock.client.collections.get.assert_called_with(
        "test_collection"
    )
    weaviate_client_mock.client.collections.query.bm25.assert_called_with(
        query=query, limit=10
    )


def test_update_object(weaviate_client_mock):
    # Test updating an object
    object_id = "12345"
    properties = {"name": "Jane"}
    weaviate_client_mock.update(
        "test_collection", object_id, properties
    )
    weaviate_client_mock.client.collections.get.assert_called_with(
        "test_collection"
    )
    weaviate_client_mock.client.collections.data.update.assert_called_with(
        object_id, properties
    )


def test_delete_object(weaviate_client_mock):
    # Test deleting an object
    object_id = "12345"
    weaviate_client_mock.delete("test_collection", object_id)
    weaviate_client_mock.client.collections.get.assert_called_with(
        "test_collection"
    )
    weaviate_client_mock.client.collections.data.delete_by_id.assert_called_with(
        object_id
    )


def test_create_collection_with_vectorizer_config(
    weaviate_client_mock,
):
    # Test creating a collection with vectorizer configuration
    vectorizer_config = {"config_key": "config_value"}
    weaviate_client_mock.create_collection(
        "test_collection", [{"name": "property"}], vectorizer_config
    )
    weaviate_client_mock.client.collections.create.assert_called_with(
        name="test_collection",
        vectorizer_config=vectorizer_config,
        properties=[{"name": "property"}],
    )


def test_query_objects_with_limit(weaviate_client_mock):
    # Test querying objects with a specified limit
    query = "name:John"
    limit = 20
    weaviate_client_mock.query("test_collection", query, limit)
    weaviate_client_mock.client.collections.get.assert_called_with(
        "test_collection"
    )
    weaviate_client_mock.client.collections.query.bm25.assert_called_with(
        query=query, limit=limit
    )


def test_query_objects_without_limit(weaviate_client_mock):
    # Test querying objects without specifying a limit
    query = "name:John"
    weaviate_client_mock.query("test_collection", query)
    weaviate_client_mock.client.collections.get.assert_called_with(
        "test_collection"
    )
    weaviate_client_mock.client.collections.query.bm25.assert_called_with(
        query=query, limit=10
    )


def test_create_collection_failure(weaviate_client_mock):
    # Test failure when creating a collection
    with patch(
        "weaviate_client.weaviate.collections.create",
        side_effect=Exception("Create error"),
    ):
        with pytest.raises(
            Exception, match="Error creating collection"
        ):
            weaviate_client_mock.create_collection(
                "test_collection", [{"name": "property"}]
            )


def test_add_object_failure(weaviate_client_mock):
    # Test failure when adding an object
    properties = {"name": "John"}
    with patch(
        "weaviate_client.weaviate.collections.data.insert",
        side_effect=Exception("Insert error"),
    ):
        with pytest.raises(Exception, match="Error adding object"):
            weaviate_client_mock.add("test_collection", properties)


def test_query_objects_failure(weaviate_client_mock):
    # Test failure when querying objects
    query = "name:John"
    with patch(
        "weaviate_client.weaviate.collections.query.bm25",
        side_effect=Exception("Query error"),
    ):
        with pytest.raises(Exception, match="Error querying objects"):
            weaviate_client_mock.query("test_collection", query)


def test_update_object_failure(weaviate_client_mock):
    # Test failure when updating an object
    object_id = "12345"
    properties = {"name": "Jane"}
    with patch(
        "weaviate_client.weaviate.collections.data.update",
        side_effect=Exception("Update error"),
    ):
        with pytest.raises(Exception, match="Error updating object"):
            weaviate_client_mock.update(
                "test_collection", object_id, properties
            )


def test_delete_object_failure(weaviate_client_mock):
    # Test failure when deleting an object
    object_id = "12345"
    with patch(
        "weaviate_client.weaviate.collections.data.delete_by_id",
        side_effect=Exception("Delete error"),
    ):
        with pytest.raises(Exception, match="Error deleting object"):
            weaviate_client_mock.delete("test_collection", object_id)
