from abc import ABC, abstractmethod
from typing import Any, Dict


class VectorDatabase(ABC):
    @abstractmethod
    def add(
        self, vector: Dict[str, Any], metadata: Dict[str, Any]
    ) -> None:
        """
        add a vector into the database.

        Args:
            vector (Dict[str, Any]): The vector to add.
            metadata (Dict[str, Any]): Metadata associated with the vector.
        """
        pass

    @abstractmethod
    def query(
        self, vector: Dict[str, Any], num_results: int
    ) -> Dict[str, Any]:
        """
        Query the database for vectors similar to the given vector.

        Args:
            vector (Dict[str, Any]): The vector to compare against.
            num_results (int): The number of similar vectors to return.

        Returns:
            Dict[str, Any]: The most similar vectors and their associated metadata.
        """
        pass

    @abstractmethod
    def delete(self, vector_id: str) -> None:
        """
        Delete a vector from the database.

        Args:
            vector_id (str): The ID of the vector to delete.
        """
        pass

    @abstractmethod
    def update(
        self,
        vector_id: str,
        vector: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> None:
        """
        Update a vector in the database.

        Args:
            vector_id (str): The ID of the vector to update.
            vector (Dict[str, Any]): The new vector.
            metadata (Dict[str, Any]): The new metadata.
        """
        pass
