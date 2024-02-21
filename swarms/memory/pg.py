import uuid
from typing import Any, List, Optional

from sqlalchemy import JSON, Column, String, create_engine
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session


class PostgresDB:
    """
    A class representing a Postgres database.

    Args:
        connection_string (str): The connection string for the Postgres database.
        table_name (str): The name of the table in the database.

    Attributes:
        engine: The SQLAlchemy engine for connecting to the database.
        table_name (str): The name of the table in the database.
        VectorModel: The SQLAlchemy model representing the vector table.

    """

    def __init__(
        self, connection_string: str, table_name: str, *args, **kwargs
    ):
        """
        Initializes a new instance of the PostgresDB class.

        Args:
            connection_string (str): The connection string for the Postgres database.
            table_name (str): The name of the table in the database.

        """
        self.engine = create_engine(
            connection_string, *args, **kwargs
        )
        self.table_name = table_name
        self.VectorModel = self._create_vector_model()

    def _create_vector_model(self):
        """
        Creates the SQLAlchemy model for the vector table.

        Returns:
            The SQLAlchemy model representing the vector table.

        """
        Base = declarative_base()

        class VectorModel(Base):
            __tablename__ = self.table_name

            id = Column(
                UUID(as_uuid=True),
                primary_key=True,
                default=uuid.uuid4,
                unique=True,
                nullable=False,
            )
            vector = Column(
                String
            )  # Assuming vector is stored as a string
            namespace = Column(String)
            meta = Column(JSON)

        return VectorModel

    def add_or_update_vector(
        self,
        vector: str,
        vector_id: Optional[str] = None,
        namespace: Optional[str] = None,
        meta: Optional[dict] = None,
    ) -> None:
        """
        Adds or updates a vector in the database.

        Args:
            vector (str): The vector to be added or updated.
            vector_id (str, optional): The ID of the vector. If not provided, a new ID will be generated.
            namespace (str, optional): The namespace of the vector.
            meta (dict, optional): Additional metadata associated with the vector.

        """
        try:
            with Session(self.engine) as session:
                obj = self.VectorModel(
                    id=vector_id,
                    vector=vector,
                    namespace=namespace,
                    meta=meta,
                )
                session.merge(obj)
                session.commit()
        except Exception as e:
            print(f"Error adding or updating vector: {e}")

    def query_vectors(
        self, query: Any, namespace: Optional[str] = None
    ) -> List[Any]:
        """
        Queries vectors from the database based on the given query and namespace.

        Args:
            query (Any): The query or condition to filter the vectors.
            namespace (str, optional): The namespace of the vectors to be queried.

        Returns:
            List[Any]: A list of vectors that match the query and namespace.

        """
        try:
            with Session(self.engine) as session:
                q = session.query(self.VectorModel)
                if namespace:
                    q = q.filter_by(namespace=namespace)
                # Assuming 'query' is a condition or filter
                q = q.filter(query)
                return q.all()
        except Exception as e:
            print(f"Error querying vectors: {e}")
            return []

    def delete_vector(self, vector_id):
        """
        Deletes a vector from the database based on the given vector ID.

        Args:
            vector_id: The ID of the vector to be deleted.

        """
        try:
            with Session(self.engine) as session:
                obj = session.get(self.VectorModel, vector_id)
                if obj:
                    session.delete(obj)
                    session.commit()
        except Exception as e:
            print(f"Error deleting vector: {e}")
