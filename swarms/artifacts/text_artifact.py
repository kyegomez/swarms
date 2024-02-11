from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field
from swarms.artifacts.base_artifact import BaseArtifact
from swarms.tokenizers.base_tokenizer import BaseTokenizer


@dataclass
class TextArtifact(BaseArtifact):
    """
    Represents a text artifact.

    Attributes:
        value (str): The text value of the artifact.
        encoding (str, optional): The encoding of the text (default is "utf-8").
        encoding_error_handler (str, optional): The error handler for encoding errors (default is "strict").
        _embedding (list[float]): The embedding of the text artifact (default is an empty list).

    Properties:
        embedding (Optional[list[float]]): The embedding of the text artifact.

    Methods:
        __add__(self, other: BaseArtifact) -> TextArtifact: Concatenates the text value of the artifact with another artifact.
        __bool__(self) -> bool: Checks if the text value of the artifact is non-empty.
        generate_embedding(self, driver: BaseEmbeddingModel) -> Optional[list[float]]: Generates the embedding of the text artifact using a given embedding model.
        token_count(self, tokenizer: BaseTokenizer) -> int: Counts the number of tokens in the text artifact using a given tokenizer.
        to_bytes(self) -> bytes: Converts the text value of the artifact to bytes using the specified encoding and error handler.
    """

    value: str
    encoding: str = "utf-8"
    encoding_error_handler: str = "strict"
    _embedding: list[float] = field(default_factory=list)

    @property
    def embedding(self) -> Optional[list[float]]:
        return None if len(self._embedding) == 0 else self._embedding

    def __add__(self, other: BaseArtifact) -> TextArtifact:
        return TextArtifact(self.value + other.value)

    def __bool__(self) -> bool:
        return bool(self.value.strip())

    def generate_embedding(self, model) -> Optional[list[float]]:
        self._embedding.clear()
        self._embedding.extend(model.embed_string(str(self.value)))

        return self.embedding

    def token_count(self, tokenizer: BaseTokenizer) -> int:
        return tokenizer.count_tokens(str(self.value))

    def to_bytes(self) -> bytes:
        return self.value.encode(
            encoding=self.encoding, errors=self.encoding_error_handler
        )
