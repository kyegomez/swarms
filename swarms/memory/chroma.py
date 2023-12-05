from typing import List
from chromadb.utils import embedding_functions
from httpx import RequestError
import chromadb


class ChromaClient:
    def __init__(
        self,
        collection_name: str = "chromadb-collection",
        model_name: str = "BAAI/bge-small-en-v1.5",
    ):
        try:
            self.client = chromadb.Client()
            self.collection_name = collection_name
            self.model = None
            self.collection = None
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
            self.model =embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        except Exception as e:
            print(f"Error loading embedding model: {e}")

    def _setup_collection(self):
        try:
            self.collection = self.client.get_collection(name=self.collection_name, embedding_function=self.model)
        except Exception as e:
            print(f"{e}. Creating new collection: {self.collection}")

        self.collection = self.client.create_collection(name=self.collection_name, embedding_function=self.model)


    def add_vectors(self, docs: List[str]):
        """
        Adds vector representations of documents to the Qdrant collection.

        Args:
            docs (List[dict]): A list of documents where each document is a dictionary with at least a 'page_content' key.

        Returns:
            OperationResponse or None: Returns the operation information if successful, otherwise None.
        """
        points = []
        ids = []
        for i, doc in enumerate(docs):
            try:
                points.append(doc)
                ids.append("id"+str(i))
            except Exception as e:
                print(f"Error processing document at index {i}: {e}")

        try:
            self.collection.add(
                documents=points,
                ids=ids
            )
        except Exception as e:
            print(f"Error adding vectors: {e}")
            return None

    def search_vectors(self, query: str, limit: int = 2):
        """
        Searches the collection for vectors similar to the query vector.

        Args:
            query (str): The query string to be converted into a vector and used for searching.
            limit (int): The number of search results to return. Defaults to 3.

        Returns:
            SearchResult or None: Returns the search results if successful, otherwise None.
        """
        try:
            search_result = self.collection.query(
                                    query_texts=query,
                                    n_results=limit,
                                )
            return search_result
        except Exception as e:
            print(f"Error searching vectors: {e}")
            return None

    def search_vectors_formatted(self, query: str, limit: int = 2):
        """
        Searches the collection for vectors similar to the query vector.

        Args:
            query (str): The query string to be converted into a vector and used for searching.
            limit (int): The number of search results to return. Defaults to 3.

        Returns:
            SearchResult or None: Returns the search results if successful, otherwise None.
        """
        try:
            search_result = self.collection.query(
                                    query_texts=query,
                                    n_results=limit,
                                )
            return search_result
        except Exception as e:
            print(f"Error searching vectors: {e}")
            return None
