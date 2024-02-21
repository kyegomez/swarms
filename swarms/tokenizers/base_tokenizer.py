from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class BaseTokenizer(ABC):
    """
    Base class for tokenizers.

    Attributes:
        stop_sequences (List[str]): List of stop sequences.
        max_tokens (int): Maximum number of tokens.
        stop_token (str): Stop token.
    """

    max_tokens: int
    stop_token: str = "<|Response|>"

    def __post_init__(self):
        self.stop_sequences: list[str] = field(
            default_factory=lambda: ["<|Response|>"],
            init=False,
        )

    def count_tokens_left(self, text: str | list[dict]) -> int:
        """
        Counts the number of tokens left based on the given text.

        Args:
            text (Union[str, List[dict]]): The text to count tokens from.

        Returns:
            int: The number of tokens left.
        """
        diff = self.max_tokens - self.count_tokens(text)

        if diff > 0:
            return diff
        else:
            return 0

    @abstractmethod
    def count_tokens(self, text: str | list[dict]) -> int:
        """
        Counts the number of tokens in the given text.

        Args:
            text (Union[str, List[dict]]): The text to count tokens from.

        Returns:
            int: The number of tokens.
        """
        ...
