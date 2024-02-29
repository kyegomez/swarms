import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class InternalMemoryBase(ABC):
    """Abstract base class for internal memory of agents in the swarm."""

    def __init__(self, n_entries):
        """Initialize the internal memory. In the current architecture the memory always consists of a set of soltuions or evaluations.
        During the operation, the agent should retrivie best solutions from it's internal memory based on the score.

        Moreover, the project is designed around LLMs for the proof of concepts, so we treat all entry content as a string.
        """
        self.n_entries = n_entries

    @abstractmethod
    def add(self, score, entry):
        """Add an entry to the internal memory."""
        raise NotImplementedError

    @abstractmethod
    def get_top_n(self, n):
        """Get the top n entries from the internal memory."""
        raise NotImplementedError


class DictInternalMemory(InternalMemoryBase):
    def __init__(self, n_entries: int):
        """
        Initialize the internal memory. In the current architecture the memory always consists of a set of solutions or evaluations.
        Simple key-value store for now.

        Args:
            n_entries (int): The maximum number of entries to keep in the internal memory.
        """
        super().__init__(n_entries)
        self.data: Dict[str, Dict[str, Any]] = {}

    def add(self, score: float, content: Any) -> None:
        """
        Add an entry to the internal memory.

        Args:
            score (float): The score or fitness value associated with the entry.
            content (Any): The content of the entry.

        Returns:
            None
        """
        random_key: str = str(uuid.uuid4())
        self.data[random_key] = {"score": score, "content": content}

        # keep only the best n entries
        sorted_data: List[Tuple[str, Dict[str, Any]]] = sorted(
            self.data.items(),
            key=lambda x: x[1]["score"],
            reverse=True,
        )
        self.data = dict(sorted_data[: self.n_entries])

    def get_top_n(self, n: int) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Get the top n entries from the internal memory.

        Args:
            n (int): The number of top entries to retrieve.

        Returns:
            List[Tuple[str, Dict[str, Any]]]: A list of tuples containing the random keys and corresponding entry data.
        """
        sorted_data: List[Tuple[str, Dict[str, Any]]] = sorted(
            self.data.items(),
            key=lambda x: x[1]["score"],
            reverse=True,
        )
        return sorted_data[:n]

    def len(self) -> int:
        """
        Get the number of entries in the internal memory.

        Returns:
            int: The number of entries in the internal memory.
        """
        return len(self.data)
