from typing import Dict, List, Optional, Any, Union
from loguru import logger
from typedb.client import TypeDB, SessionType, TransactionType
from typedb.api.connection.transaction import Transaction
from dataclasses import dataclass
import json

@dataclass
class TypeDBConfig:
    """Configuration for TypeDB connection."""
    uri: str = "localhost:1729"
    database: str = "swarms"
    username: Optional[str] = None
    password: Optional[str] = None
    timeout: int = 30

class TypeDBWrapper:
    """
    A wrapper class for TypeDB that provides graph database operations for Swarms.
    This class handles connection, schema management, and data operations.
    """

    def __init__(self, config: Optional[TypeDBConfig] = None):
        """
        Initialize the TypeDB wrapper with the given configuration.
        Args:
            config (Optional[TypeDBConfig]): Configuration for TypeDB connection.
        """
        self.config = config or TypeDBConfig()
        self.client = None
        self.session = None
        self._connect()

    def _connect(self) -> None:
        """Establish connection to TypeDB."""
        try:
            self.client = TypeDB.core_client(self.config.uri)
            if self.config.username and self.config.password:
                self.session = self.client.session(
                    self.config.database,
                    SessionType.DATA,
                    self.config.username,
                    self.config.password
                )
            else:
                self.session = self.client.session(
                    self.config.database,
                    SessionType.DATA
                )
            logger.info(f"Connected to TypeDB at {self.config.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to TypeDB: {e}")
            raise

    def _ensure_connection(self) -> None:
        """Ensure connection is active, reconnect if necessary."""
        if not self.session or not self.session.is_open():
            self._connect()

    def define_schema(self, schema: str) -> None:
        """
        Define the database schema.
        Args:
            schema (str): TypeQL schema definition.
        """
        try:
            with self.session.transaction(TransactionType.WRITE) as transaction:
                transaction.query.define(schema)
                transaction.commit()
            logger.info("Schema defined successfully")
        except Exception as e:
            logger.error(f"Failed to define schema: {e}")
            raise

    def insert_data(self, query: str) -> None:
        """
        Insert data using TypeQL query.
        Args:
            query (str): TypeQL insert query.
        """
        try:
            with self.session.transaction(TransactionType.WRITE) as transaction:
                transaction.query.insert(query)
                transaction.commit()
            logger.info("Data inserted successfully")
        except Exception as e:
            logger.error(f"Failed to insert data: {e}")
            raise

    def query_data(self, query: str) -> List[Dict[str, Any]]:
        """
        Query data using TypeQL query.
        Args:
            query (str): TypeQL query.
        Returns:
            List[Dict[str, Any]]: Query results.
        """
        try:
            with self.session.transaction(TransactionType.READ) as transaction:
                result = transaction.query.get(query)
                return [self._convert_concept_to_dict(concept) for concept in result]
        except Exception as e:
            logger.error(f"Failed to query data: {e}")
            raise

    def _convert_concept_to_dict(self, concept: Any) -> Dict[str, Any]:
        """
        Convert a TypeDB concept to a dictionary.
        Args:
            concept: TypeDB concept.
        Returns:
            Dict[str, Any]: Dictionary representation of the concept.
        """
        try:
            if hasattr(concept, "get_type"):
                concept_type = concept.get_type()
                if hasattr(concept, "get_value"):
                    return {
                        "type": concept_type.get_label_name(),
                        "value": concept.get_value()
                    }
                elif hasattr(concept, "get_attributes"):
                    return {
                        "type": concept_type.get_label_name(),
                        "attributes": {
                            attr.get_type().get_label_name(): attr.get_value()
                            for attr in concept.get_attributes()
                        }
                    }
            return {"type": "unknown", "value": str(concept)}
        except Exception as e:
            logger.error(f"Failed to convert concept to dict: {e}")
            return {"type": "error", "value": str(e)}

    def delete_data(self, query: str) -> None:
        """
        Delete data using TypeQL query.
        Args:
            query (str): TypeQL delete query.
        """
        try:
            with self.session.transaction(TransactionType.WRITE) as transaction:
                transaction.query.delete(query)
                transaction.commit()
            logger.info("Data deleted successfully")
        except Exception as e:
            logger.error(f"Failed to delete data: {e}")
            raise

    def close(self) -> None:
        """Close the TypeDB connection."""
        try:
            if self.session:
                self.session.close()
            if self.client:
                self.client.close()
            logger.info("TypeDB connection closed")
        except Exception as e:
            logger.error(f"Failed to close TypeDB connection: {e}")
            raise

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close() 