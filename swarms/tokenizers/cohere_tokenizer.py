from __future__ import annotations

from dataclasses import dataclass
from cohere import Client


@dataclass
class CohereTokenizer:
    """
    A tokenizer class for Cohere models.
    """

    model: str
    client: Client
    DEFAULT_MODEL: str = "command"
    DEFAULT_MAX_TOKENS: int = 2048
    max_tokens: int = DEFAULT_MAX_TOKENS

    def count_tokens(self, text: str | list) -> int:
        """
        Count the number of tokens in the given text.

        Args:
            text (str | list): The input text to tokenize.

        Returns:
            int: The number of tokens in the text.

        Raises:
            ValueError: If the input text is not a string.
        """
        if isinstance(text, str):
            return len(self.client.tokenize(text=text).tokens)
        else:
            raise ValueError("Text must be a string.")
