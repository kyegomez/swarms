from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Optional

from anthropic import Anthropic

from swarms.tokenizers.base_tokenizer import BaseTokenizer

INSTALL_MAPPING = {
    "huggingface_hub": "huggingface-hub",
    "pinecone": "pinecone-client",
    "opensearchpy": "opensearch-py",
}


def import_optional_dependency(name: str) -> Optional[ModuleType]:
    """Import an optional dependency.

    If a dependency is missing, an ImportError with a nice message will be raised.

    Args:
        name: The module name.
    Returns:
        The imported module, when found.
        None is returned when the package is not found and `errors` is False.
    """

    package_name = INSTALL_MAPPING.get(name)
    install_name = package_name if package_name is not None else name

    msg = (
        f"Missing optional dependency: '{install_name}'. "
        f"Use poetry or pip to install '{install_name}'."
    )
    try:
        module = import_module(name)
    except ImportError:
        raise ImportError(msg)

    return module


@dataclass
class AnthropicTokenizer(BaseTokenizer):
    """
    Tokenizer class for Anthropic models.]
    """

    max_tokens: int = 500
    client: Anthropic = None
    model: str = "claude-2.1"

    def __post_init__(self):
        self.DEFAULT_MODEL: str = "claude-2.1"
        self.MODEL_PREFIXES_TO_MAX_TOKENS: dict[str, int] = {
            "claude-2.1": 200000,
            "claude": 100000,
        }
        self.model = self.model  # or self.DEFAULT_MODEL
        self.max_tokens = self.max_tokens or self.default_max_tokens()
        self.client = (
            self.client
            or import_optional_dependency("anthropic").Anthropic()
        )

    def default_max_tokens(self) -> int:
        """
        Returns the default maximum number of tokens based on the model prefix.
        """
        tokens = next(
            v
            for k, v in self.MODEL_PREFIXES_TO_MAX_TOKENS.items()
            if self.model.startswith(k)
        )
        return tokens

    def count_tokens(self, text: str | list) -> int:
        """
        Counts the number of tokens in the given text.

        Args:
            text: The input text.

        Returns:
            The number of tokens in the text.

        Raises:
            ValueError: If the input text is not a string.
        """
        if isinstance(text, str):
            return self.client.count_tokens(text)
        else:
            raise ValueError("Text must be a string.")
