import uuid
from typing import Optional
from attr import define, field, Factory
from dataclasses import dataclass
from swarms.memory.base import BaseVectorStore
from sqlalchemy.engine import Engine
from sqlalchemy import create_engine, Column, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector


@define
class PgVectorVectorStore(BaseVectorStore):
    """A vector store driver to Postgres using the PGVector extension.

    Attributes:
        connection_string: An optional string describing the target Postgres database instance.
        create_engine_params: Additional configuration params passed when creating the database connection.
        engine: An optional sqlalchemy Postgres engine to use.
        table_name: Optionally specify the name of the table to used to store vectors.

    Methods:
        upsert_vector(vector: list[float], vector_id: Optional[str] = None, namespace: Optional[str] = None, meta: Optional[dict] = None, **kwargs) -> str:
            Upserts a vector into the index.
        load_entry(vector_id: str, namespace: Optional[str] = None) -> Optional[BaseVector.Entry]:
            Loads a single vector from the index.
        load_entries(namespace: Optional[str] = None) -> list[BaseVector.Entry]:
            Loads all vectors from the index.
        query(query: str, count: Optional[int] = None, namespace: Optional[str] = None, include_vectors: bool = False, include_metadata=True, **kwargs) -> list[BaseVector.QueryResult]:
            Queries the index for vectors similar to the given query string.
        setup(create_schema: bool = True, install_uuid_extension: bool = True, install_vector_extension: bool = True) -> None:
            Provides a mechanism to initialize the database schema and extensions.

    Usage:
    >>> from swarms.memory.vector_stores.pgvector import PgVectorVectorStore
    >>> from swarms.utils.embeddings import USEEmbedding
    >>> from swarms.utils.hash import str_to_hash
    >>> from swarms.utils.dataframe import dataframe_to_hash
    >>> import pandas as pd
    >>>
    >>> # Create a new PgVectorVectorStore instance:
    >>> pv = PgVectorVectorStore(
    >>>     connection_string="postgresql://postgres:password@localhost:5432/postgres",
    >>>     table_name="your-table-name"
    >>> )
    >>> # Create a new index:
    >>> pv.setup()
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

    connection_string: Optional[str] = field(
        default=None, kw_only=True
    )
    create_engine_params: dict = field(factory=dict, kw_only=True)
    engine: Optional[Engine] = field(default=None, kw_only=True)
    table_name: str = field(kw_only=True)
    _model: any = field(
        default=Factory(
            lambda self: self.default_vector_model(), takes_self=True
        )
    )

    @connection_string.validator
    def validate_connection_string(
        self, _, connection_string: Optional[str]
    ) -> None:
        # If an engine is provided, the connection string is not used.
        if self.engine is not None:
            return

        # If an engine is not provided, a connection string is required.
        if connection_string is None:
            raise ValueError(
                "An engine or connection string is required"
            )

        if not connection_string.startswith("postgresql://"):
            raise ValueError(
                "The connection string must describe a Postgres"
                " database connection"
            )

    @engine.validator
    def validate_engine(self, _, engine: Optional[Engine]) -> None:
        # If a connection string is provided, an engine does not need to be provided.
        if self.connection_string is not None:
            return

        # If a connection string is not provided, an engine is required.
        if engine is None:
            raise ValueError(
                "An engine or connection string is required"
            )

    def __attrs_post_init__(self) -> None:
        """If a an engine is provided, it will be used to connect to the database.
        If not, a connection string is used to create a new database connection here.
        """
        if self.engine is None:
            self.engine = create_engine(
                self.connection_string, **self.create_engine_params
            )

    def setup(
        self,
        create_schema: bool = True,
        install_uuid_extension: bool = True,
        install_vector_extension: bool = True,
    ) -> None:
        """Provides a mechanism to initialize the database schema and extensions."""
        if install_uuid_extension:
            self.engine.execute(
                'CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'
            )

        if install_vector_extension:
            self.engine.execute(
                'CREATE EXTENSION IF NOT EXISTS "vector";'
            )

        if create_schema:
            self._model.metadata.create_all(self.engine)

    def upsert_vector(
        self,
        vector: list[float],
        vector_id: Optional[str] = None,
        namespace: Optional[str] = None,
        meta: Optional[dict] = None,
        **kwargs,
    ) -> str:
        """Inserts or updates a vector in the collection."""
        with Session(self.engine) as session:
            obj = self._model(
                id=vector_id,
                vector=vector,
                namespace=namespace,
                meta=meta,
            )

            obj = session.merge(obj)
            session.commit()

            return str(obj.id)

    def load_entry(
        self, vector_id: str, namespace: Optional[str] = None
    ) -> BaseVectorStore.Entry:
        """Retrieves a specific vector entry from the collection based on its identifier and optional namespace."""
        with Session(self.engine) as session:
            result = session.get(self._model, vector_id)

            return BaseVectorStore.Entry(
                id=result.id,
                vector=result.vector,
                namespace=result.namespace,
                meta=result.meta,
            )

    def load_entries(
        self, namespace: Optional[str] = None
    ) -> list[BaseVectorStore.Entry]:
        """Retrieves all vector entries from the collection, optionally filtering to only
        those that match the provided namespace.
        """
        with Session(self.engine) as session:
            query = session.query(self._model)
            if namespace:
                query = query.filter_by(namespace=namespace)

            results = query.all()

            return [
                BaseVectorStore.Entry(
                    id=str(result.id),
                    vector=result.vector,
                    namespace=result.namespace,
                    meta=result.meta,
                )
                for result in results
            ]

    def query(
        self,
        query: str,
        count: Optional[int] = BaseVectorStore.DEFAULT_QUERY_COUNT,
        namespace: Optional[str] = None,
        include_vectors: bool = False,
        distance_metric: str = "cosine_distance",
        **kwargs,
    ) -> list[BaseVectorStore.QueryResult]:
        """Performs a search on the collection to find vectors similar to the provided input vector,
        optionally filtering to only those that match the provided namespace.
        """
        distance_metrics = {
            "cosine_distance": self._model.vector.cosine_distance,
            "l2_distance": self._model.vector.l2_distance,
            "inner_product": self._model.vector.max_inner_product,
        }

        if distance_metric not in distance_metrics:
            raise ValueError("Invalid distance metric provided")

        op = distance_metrics[distance_metric]

        with Session(self.engine) as session:
            vector = self.embedding_driver.embed_string(query)

            # The query should return both the vector and the distance metric score.
            query = session.query(
                self._model,
                op(vector).label("score"),
            ).order_by(op(vector))

            if namespace:
                query = query.filter_by(namespace=namespace)

            results = query.limit(count).all()

            return [
                BaseVectorStore.QueryResult(
                    id=str(result[0].id),
                    vector=(
                        result[0].vector if include_vectors else None
                    ),
                    score=result[1],
                    meta=result[0].meta,
                    namespace=result[0].namespace,
                )
                for result in results
            ]

    def default_vector_model(self) -> any:
        Base = declarative_base()

        @dataclass
        class VectorModel(Base):
            __tablename__ = self.table_name

            id = Column(
                UUID(as_uuid=True),
                primary_key=True,
                default=uuid.uuid4,
                unique=True,
                nullable=False,
            )
            vector = Column(Vector())
            namespace = Column(String)
            meta = Column(JSON)

        return VectorModel
