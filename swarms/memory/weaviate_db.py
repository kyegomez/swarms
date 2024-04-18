"""
Weaviate API Client
"""

from typing import Any, Dict, List, Optional

from swarms.memory.base_vectordb import AbstractVectorDatabase

try:
    import weaviate
except ImportError:
    print("pip install weaviate-client")


class WeaviateDB(AbstractVectorDatabase):
    """

    Weaviate API Client
    Interface to Weaviate, a vector database with a GraphQL API.

    Args:
        http_host (str): The HTTP host of the Weaviate server.
        http_port (str): The HTTP port of the Weaviate server.
        http_secure (bool): Whether to use HTTPS.
        grpc_host (Optional[str]): The gRPC host of the Weaviate server.
        grpc_port (Optional[str]): The gRPC port of the Weaviate server.
        grpc_secure (Optional[bool]): Whether to use gRPC over TLS.
        auth_client_secret (Optional[Any]): The authentication client secret.
        additional_headers (Optional[Dict[str, str]]): Additional headers to send with requests.
        additional_config (Optional[weaviate.AdditionalConfig]): Additional configuration for the client.

    Methods:
        create_collection: Create a new collection in Weaviate.
        add: Add an object to a specified collection.
        query: Query objects from a specified collection.
        update: Update an object in a specified collection.
        delete: Delete an object from a specified collection.

    Examples:
    >>> from swarms.memory import WeaviateDB
    """

    def __init__(
        self,
        http_host: str,
        http_port: str,
        http_secure: bool,
        grpc_host: Optional[str] = None,
        grpc_port: Optional[str] = None,
        grpc_secure: Optional[bool] = None,
        auth_client_secret: Optional[Any] = None,
        additional_headers: Optional[Dict[str, str]] = None,
        additional_config: Optional[Any] = None,
        connection_params: Dict[str, Any] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.http_host = http_host
        self.http_port = http_port
        self.http_secure = http_secure
        self.grpc_host = grpc_host
        self.grpc_port = grpc_port
        self.grpc_secure = grpc_secure
        self.auth_client_secret = auth_client_secret
        self.additional_headers = additional_headers
        self.additional_config = additional_config
        self.connection_params = connection_params

        # If connection_params are provided, use them to initialize the client.
        connection_params = weaviate.ConnectionParams.from_params(
            http_host=http_host,
            http_port=http_port,
            http_secure=http_secure,
            grpc_host=grpc_host,
            grpc_port=grpc_port,
            grpc_secure=grpc_secure,
        )

        # If additional headers are provided, add them to the connection params.
        self.client = weaviate.WeaviateDB(
            connection_params=connection_params,
            auth_client_secret=auth_client_secret,
            additional_headers=additional_headers,
            additional_config=additional_config,
        )

    def create_collection(
        self,
        name: str,
        properties: List[Dict[str, Any]],
        vectorizer_config: Any = None,
    ):
        """Create a new collection in Weaviate.

        Args:
            name (str): _description_
            properties (List[Dict[str, Any]]): _description_
            vectorizer_config (Any, optional): _description_. Defaults to None.
        """
        try:
            out = self.client.collections.create(
                name=name,
                vectorizer_config=vectorizer_config,
                properties=properties,
            )
            print(out)
        except Exception as error:
            print(f"Error creating collection: {error}")
            raise

    def add(self, collection_name: str, properties: Dict[str, Any]):
        """Add an object to a specified collection.

        Args:
            collection_name (str): _description_
            properties (Dict[str, Any]): _description_

        Returns:
            _type_: _description_
        """
        try:
            collection = self.client.collections.get(collection_name)
            return collection.data.insert(properties)
        except Exception as error:
            print(f"Error adding object: {error}")
            raise

    def query(self, collection_name: str, query: str, limit: int = 10):
        """Query objects from a specified collection.

        Args:
            collection_name (str): _description_
            query (str): _description_
            limit (int, optional): _description_. Defaults to 10.

        Returns:
            _type_: _description_
        """
        try:
            collection = self.client.collections.get(collection_name)
            response = collection.query.bm25(query=query, limit=limit)
            return [o.properties for o in response.objects]
        except Exception as error:
            print(f"Error querying objects: {error}")
            raise

    def update(
        self,
        collection_name: str,
        object_id: str,
        properties: Dict[str, Any],
    ):
        """UPdate an object in a specified collection.

        Args:
            collection_name (str): _description_
            object_id (str): _description_
            properties (Dict[str, Any]): _description_
        """
        try:
            collection = self.client.collections.get(collection_name)
            collection.data.update(object_id, properties)
        except Exception as error:
            print(f"Error updating object: {error}")
            raise

    def delete(self, collection_name: str, object_id: str):
        """Delete an object from a specified collection.

        Args:
            collection_name (str): _description_
            object_id (str): _description_
        """
        try:
            collection = self.client.collections.get(collection_name)
            collection.data.delete_by_id(object_id)
        except Exception as error:
            print(f"Error deleting object: {error}")
            raise
