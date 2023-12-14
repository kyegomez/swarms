import subprocess
from typing import List
from httpx import RequestError

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install the sentence-transformers package")
    print("pip install sentence-transformers")
    print("pip install qdrant-client")
    subprocess.run(["pip", "install", "sentence-transformers"])


try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import (
        Distance,
        VectorParams,
        PointStruct,
    )
except ImportError:
    print("Please install the qdrant-client package")
    print("pip install qdrant-client")
    subprocess.run(["pip", "install", "qdrant-client"])


class Qdrant:
    """
    Qdrant class for managing collections and performing vector operations using QdrantClient.

    Attributes:
        client (QdrantClient): The Qdrant client for interacting with the Qdrant server.
        collection_name (str): Name of the collection to be managed in Qdrant.
        model (SentenceTransformer): The model used for generating sentence embeddings.

    Args:
        api_key (str): API key for authenticating with Qdrant.
        host (str): Host address of the Qdrant server.
        port (int): Port number of the Qdrant server. Defaults to 6333.
        collection_name (str): Name of the collection to be used or created. Defaults to "qdrant".
        model_name (str): Name of the model to be used for embeddings. Defaults to "BAAI/bge-small-en-v1.5".
        https (bool): Flag to indicate if HTTPS should be used. Defaults to True.
    """

    def __init__(
        self,
        api_key: str,
        host: str,
        port: int = 6333,
        collection_name: str = "qdrant",
        model_name: str = "BAAI/bge-small-en-v1.5",
        https: bool = True,
    ):
        try:
            self.client = QdrantClient(
                url=host, port=port, api_key=api_key
            )
            self.collection_name = collection_name
            self._load_embedding_model(model_name)
            self._setup_collection()
        except RequestError as e:
            print(f"Error setting up QdrantClient: {e}")

    def _load_embedding_model(self, model_name: str):
        """
        Loads the sentence embedding model specified by the model name.

        Args:
            model_name (str): The name of the model to load for generating embeddings.
        """
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error loading embedding model: {e}")

    def _setup_collection(self):
        try:
            exists = self.client.get_collection(self.collection_name)
            if exists:
                print(
                    f"Collection '{self.collection_name}' already"
                    " exists."
                )
        except Exception as e:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.model.get_sentence_embedding_dimension(),
                    distance=Distance.DOT,
                ),
            )
            print(f"Collection '{self.collection_name}' created.")

    def add_vectors(self, docs: List[dict]):
        """
        Adds vector representations of documents to the Qdrant collection.

        Args:
            docs (List[dict]): A list of documents where each document is a dictionary with at least a 'page_content' key.

        Returns:
            OperationResponse or None: Returns the operation information if successful, otherwise None.
        """
        points = []
        for i, doc in enumerate(docs):
            try:
                if "page_content" in doc:
                    embedding = self.model.encode(
                        doc["page_content"], normalize_embeddings=True
                    )
                    points.append(
                        PointStruct(
                            id=i + 1,
                            vector=embedding,
                            payload={"content": doc["page_content"]},
                        )
                    )
                else:
                    print(
                        f"Document at index {i} is missing"
                        " 'page_content' key"
                    )
            except Exception as e:
                print(f"Error processing document at index {i}: {e}")

        try:
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=points,
            )
            return operation_info
        except Exception as e:
            print(f"Error adding vectors: {e}")
            return None

    def search_vectors(self, query: str, limit: int = 3):
        """
        Searches the collection for vectors similar to the query vector.

        Args:
            query (str): The query string to be converted into a vector and used for searching.
            limit (int): The number of search results to return. Defaults to 3.

        Returns:
            SearchResult or None: Returns the search results if successful, otherwise None.
        """
        try:
            query_vector = self.model.encode(
                query, normalize_embeddings=True
            )
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
            )
            return search_result
        except Exception as e:
            print(f"Error searching vectors: {e}")
            return None
