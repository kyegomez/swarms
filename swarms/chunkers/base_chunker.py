from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field

from swarms.artifacts.text_artifact import TextArtifact
from swarms.chunkers.chunk_seperator import ChunkSeparator
from swarms.tokenizers.base_tokenizer import BaseTokenizer
from swarms.tokenizers.openai_tokenizers import OpenAITokenizer


@dataclass
class BaseChunker(ABC):
    """
    Base class for chunking text into smaller chunks.
    """

    DEFAULT_SEPARATORS = [ChunkSeparator(" ")]

    separators: list[ChunkSeparator] = field(
        default_factory=lambda: BaseChunker.DEFAULT_SEPARATORS
    )
    tokenizer: BaseTokenizer = field(
        default_factory=lambda: OpenAITokenizer(
            model=OpenAITokenizer.DEFAULT_OPENAI_GPT_3_CHAT_MODEL
        )
    )
    max_tokens: int = field(
        default_factory=lambda: BaseChunker.tokenizer.max_tokens
    )

    def chunk(self, text: str | str) -> list[str]:
        """
        Chunk the given text into smaller chunks.

        Args:
            text (TextArtifact | str): The text to be chunked.

        Returns:
            list[TextArtifact]: The list of chunked text artifacts.
        """
        text = text.value if isinstance(text, str) else text

        return [
            TextArtifact(c) for c in self._chunk_recursively(text)
        ]

    def _chunk_recursively(
        self,
        chunk: str,
        current_separator: ChunkSeparator | None = None,
    ) -> list[str]:
        """
        Recursively chunk the given chunk into smaller subchunks.

        Args:
            chunk (str): The chunk to be recursively chunked.
            current_separator (Optional[ChunkSeparator], optional): The current separator to be used. Defaults to None.

        Returns:
            list[str]: The list of recursively chunked subchunks.
        """
        token_count = self.tokenizer.count_tokens(chunk)

        if token_count <= self.max_tokens:
            return [chunk]
        else:
            balance_index = -1
            balance_diff = float("inf")
            tokens_count = 0
            half_token_count = token_count // 2

            # If a separator is provided, only use separators after it.
            if current_separator:
                separators = self.separators[
                    self.separators.index(current_separator) :
                ]
            else:
                separators = self.separators

            # Loop through available separators to find the best split.
            for separator in separators:
                # Split the chunk into subchunks using the current separator.
                subchunks = list(
                    filter(None, chunk.split(separator.value))
                )

                # Check if the split resulted in more than one subchunk.
                if len(subchunks) > 1:
                    # Iterate through the subchunks and calculate token counts.
                    for index, subchunk in enumerate(subchunks):
                        if index < len(subchunks):
                            if separator.is_prefix:
                                subchunk = separator.value + subchunk
                            else:
                                subchunk = subchunk + separator.value

                        tokens_count += self.tokenizer.count_tokens(
                            subchunk
                        )

                        # Update the best split if the current one is more balanced.
                        if (
                            abs(tokens_count - half_token_count)
                            < balance_diff
                        ):
                            balance_index = index
                            balance_diff = abs(
                                tokens_count - half_token_count
                            )

                    # Create the two subchunks based on the best separator.
                    if separator.is_prefix:
                        # If the separator is a prefix, append it before this subchunk.
                        first_subchunk = (
                            separator.value
                            + separator.value.join(
                                subchunks[: balance_index + 1]
                            )
                        )
                        second_subchunk = (
                            separator.value
                            + separator.value.join(
                                subchunks[balance_index + 1 :]
                            )
                        )
                    else:
                        # If the separator is not a prefix, append it after this subchunk.
                        first_subchunk = (
                            separator.value.join(
                                subchunks[: balance_index + 1]
                            )
                            + separator.value
                        )
                        second_subchunk = separator.value.join(
                            subchunks[balance_index + 1 :]
                        )

                    # Continue recursively chunking the subchunks.
                    first_subchunk_rec = self._chunk_recursively(
                        first_subchunk.strip(), separator
                    )
                    second_subchunk_rec = self._chunk_recursively(
                        second_subchunk.strip(), separator
                    )

                    # Return the concatenated results of the subchunks if both are non-empty.
                    if first_subchunk_rec and second_subchunk_rec:
                        return (
                            first_subchunk_rec + second_subchunk_rec
                        )
                    # If only one subchunk is non-empty, return it.
                    elif first_subchunk_rec:
                        return first_subchunk_rec
                    elif second_subchunk_rec:
                        return second_subchunk_rec
                    else:
                        return []
            # If none of the separators result in a balanced split, split the chunk in half.
            midpoint = len(chunk) // 2
            return self._chunk_recursively(
                chunk[:midpoint]
            ) + self._chunk_recursively(chunk[midpoint:])
