from typing import Optional

import pinecone
from attr import define, field

from swarms.memory.base_vectordb import BaseVectorDatabase
from swarms.utils.hash import str_to_hash


@define
class PineconeDB(BaseVectorDatabase):
    """
    PineconeDB is a vector storage driver that uses Pinecone as the underlying storage engine.

    Pinecone is a vector database that allows you to store, search, and retrieve high-dimensional vectors with
    blazing speed and low latency. It is a managed service that is easy to use and scales effortlessly, so you can
    focus on building your applications instead of managing your infrastructure.

    Args:
        api_key (str): The API key for your Pinecone account.
        index_name (str): The name of the index to use.
        environment (str): The environment to use. Either "us-west1-gcp" or "us-east1-gcp".
        project_name (str, optional): The name of the project to use. Defaults to None.
        index (pinecone.Index, optional): The Pinecone index to use. Defaults to None.

    Methods:
        upsert_vector(vector: list[float], vector_id: Optional[str] = None, namespace: Optional[str] = None, meta: Optional[dict] = None, **kwargs) -> str:
            Upserts a vector into the index.
        load_entry(vector_id: str, namespace: Optional[str] = None) -> Optional[BaseVectorStore.Entry]:
            Loads a single vector from the index.
        load_entries(namespace: Optional[str] = None) -> list[BaseVectorStore.Entry]:
            Loads all vectors from the index.
        query(query: str, count: Optional[int] = None, namespace: Optional[str] = None, include_vectors: bool = False, include_metadata=True, **kwargs) -> list[BaseVectorStore.QueryResult]:
            Queries the index for vectors similar to the given query string.
        create_index(name: str, **kwargs) -> None:
            Creates a new index.

    Usage:
    >>> from swarms.memory.vector_stores.pinecone import PineconeDB
    >>> from swarms.utils.embeddings import USEEmbedding
    >>> from swarms.utils.hash import str_to_hash
    >>> from swarms.utils.dataframe import dataframe_to_hash
    >>> import pandas as pd
    >>>
    >>> # Create a new PineconeDB instance:
    >>> pv = PineconeDB(
    >>>     api_key="your-api-key",
    >>>     index_name="your-index-name",
    >>>     environment="us-west1-gcp",
    >>>     project_name="your-project-name"
    >>> )
    >>> # Create a new index:
    >>> pv.create_index("your-index-name")
    >>> # Create a new USEEmbedding instance:
    >>> use = USEEmbedding()
    >>> # Create a new dataframe:
    >>> df = pd.DataFrame({
    >>>     "text": [
    >>>         "This is a test",
    >>>         "This is another test",
    >>>         "This is a third test"
    >>>     ]
    >>> })
    >>> # Embed the dataframe:
    >>> df["embedding"] = df["text"].apply(use.embed_string)
    >>> # Upsert the dataframe into the index:
    >>> pv.upsert_vector(
    >>>     vector=df["embedding"].tolist(),
    >>>     vector_id=dataframe_to_hash(df),
    >>>     namespace="your-namespace"
    >>> )
    >>> # Query the index:
    >>> pv.query(
    >>>     query="This is a test",
    >>>     count=10,
    >>>     namespace="your-namespace"
    >>> )
    >>> # Load a single entry from the index:
    >>> pv.load_entry(
    >>>     vector_id=dataframe_to_hash(df),
    >>>     namespace="your-namespace"
    >>> )
    >>> # Load all entries from the index:
    >>> pv.load_entries(
    >>>     namespace="your-namespace"
    >>> )


    """

    api_key: str = field(kw_only=True)
    index_name: str = field(kw_only=True)
    environment: str = field(kw_only=True)
    project_name: Optional[str] = field(default=None, kw_only=True)
    index: pinecone.Index = field(init=False)

    def __attrs_post_init__(self) -> None:
        """Post init"""
        pinecone.init(
            api_key=self.api_key,
            environment=self.environment,
            project_name=self.project_name,
        )

        self.index = pinecone.Index(self.index_name)

    def add(
        self,
        vector: list[float],
        vector_id: Optional[str] = None,
        namespace: Optional[str] = None,
        meta: Optional[dict] = None,
        **kwargs,
    ) -> str:
        """Add a vector to the index.

        Args:
            vector (list[float]): _description_
            vector_id (Optional[str], optional): _description_. Defaults to None.
            namespace (Optional[str], optional): _description_. Defaults to None.
            meta (Optional[dict], optional): _description_. Defaults to None.

        Returns:
            str: _description_
        """
        vector_id = vector_id if vector_id else str_to_hash(str(vector))

        params = {"namespace": namespace} | kwargs

        self.index.upsert([(vector_id, vector, meta)], **params)

        return vector_id

    def load_entries(self, namespace: Optional[str] = None):
        """Load all entries from the index.

        Args:
            namespace (Optional[str], optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # This is a hacky way to query up to 10,000 values from Pinecone. Waiting on an official API for fetching
        # all values from a namespace:
        # https://community.pinecone.io/t/is-there-a-way-to-query-all-the-vectors-and-or-metadata-from-a-namespace/797/5

        results = self.index.query(
            self.embedding_driver.embed_string(""),
            top_k=10000,
            include_metadata=True,
            namespace=namespace,
        )

        for result in results["matches"]:
            entry = {
                "id": result["id"],
                "vector": result["values"],
                "meta": result["metadata"],
                "namespace": result["namespace"],
            }
            return entry

    def query(
        self,
        query: str,
        count: Optional[int] = None,
        namespace: Optional[str] = None,
        include_vectors: bool = False,
        # PineconeDBStorageDriver-specific params:
        include_metadata=True,
        **kwargs,
    ):
        """Query the index for vectors similar to the given query string.

        Args:
            query (str): _description_
            count (Optional[int], optional): _description_. Defaults to None.
            namespace (Optional[str], optional): _description_. Defaults to None.
            include_vectors (bool, optional): _description_. Defaults to False.
            include_metadata (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        vector = self.embedding_driver.embed_string(query)

        params = {
            "top_k": count,
            "namespace": namespace,
            "include_values": include_vectors,
            "include_metadata": include_metadata,
        } | kwargs

        results = self.index.query(vector, **params)

        for r in results["matches"]:
            entry = {
                "id": results["id"],
                "vector": results["values"],
                "score": results["scores"],
                "meta": results["metadata"],
                "namespace": results["namespace"],
            }
            return entry

    def create_index(self, name: str, **kwargs) -> None:
        """Create a new index.

        Args:
            name (str): _description_
        """
        params = {
            "name": name,
            "dimension": self.embedding_driver.dimensions,
        } | kwargs

        pinecone.create_index(**params)
