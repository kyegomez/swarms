from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from typing import Callable
from swarms.artifacts.text_artifact import TextArtifact
from swarms.utils.exponential_backoff import ExponentialBackoffMixin


@dataclass
class BaseEmbeddingModel(
    ExponentialBackoffMixin,
    ABC,
    # SerializableMixin
):
    """
    Attributes:
        model: The name of the model to use.
        tokenizer: An instance of `BaseTokenizer` to use when calculating tokens.
    """

    model: str = None
    tokenizer: Callable = None
    chunker: Callable = None

    def embed_text_artifact(self, artifact: TextArtifact) -> list[float]:
        return self.embed_string(artifact.to_text())

    def embed_string(self, string: str) -> list[float]:
        for attempt in self.retrying():
            with attempt:
                if (
                    self.tokenizer
                    and self.tokenizer.count_tokens(string)
                    > self.tokenizer.max_tokens
                ):
                    return self._embed_long_string(string)
                else:
                    return self.try_embed_chunk(string)

        else:
            raise RuntimeError("Failed to embed string.")

    @abstractmethod
    def try_embed_chunk(self, chunk: str) -> list[float]: ...

    def _embed_long_string(self, string: str) -> list[float]:
        """Embeds a string that is too long to embed in one go."""
        chunks = self.chunker.chunk(string)

        embedding_chunks = []
        length_chunks = []
        for chunk in chunks:
            embedding_chunks.append(self.try_embed_chunk(chunk.value))
            length_chunks.append(len(chunk))

        # generate weighted averages
        embedding_chunks = np.average(
            embedding_chunks, axis=0, weights=length_chunks
        )

        # normalize length to 1
        embedding_chunks = embedding_chunks / np.linalg.norm(
            embedding_chunks
        )

        return embedding_chunks.tolist()
