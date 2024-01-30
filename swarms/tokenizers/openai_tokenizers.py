from __future__ import annotations
import logging
from dataclasses import dataclass, field
import tiktoken
from tiktoken import Encoding
from typing import Optional
from swarms.tokenizers.base_tokenizer import BaseTokenizer


@dataclass
class OpenAITokenizer(BaseTokenizer):
    """
    A class representing an OpenAI tokenizer.

    Attributes:
    - DEFAULT_OPENAI_GPT_3_COMPLETION_MODEL (str): The default OpenAI GPT-3 completion model.
    - DEFAULT_OPENAI_GPT_3_CHAT_MODEL (str): The default OpenAI GPT-3 chat model.
    - DEFAULT_OPENAI_GPT_4_MODEL (str): The default OpenAI GPT-4 model.
    - DEFAULT_ENCODING (str): The default encoding.
    - DEFAULT_MAX_TOKENS (int): The default maximum number of tokens.
    - TOKEN_OFFSET (int): The token offset.
    - MODEL_PREFIXES_TO_MAX_TOKENS (dict): A dictionary mapping model prefixes to maximum tokens.
    - EMBEDDING_MODELS (list): A list of embedding models.
    - model (str): The model name.

    Methods:
    - __post_init__(): Initializes the OpenAITokenizer object.
    - encoding(): Returns the encoding for the model.
    - default_max_tokens(): Returns the default maximum number of tokens.
    - count_tokens(text, model): Counts the number of tokens in the given text.
    - len(text, model): Returns the length of the text in tokens.
    """

    model: str = "gpt-2"

    def __post_init__(self):
        """
        Initializes the OpenAITokenizer object.
        Sets the default maximum number of tokens.
        """
        self.max_tokens: int = field(
            default_factory=lambda: self.default_max_tokens()
        )

        self.DEFAULT_OPENAI_GPT_3_COMPLETION_MODEL = (
            "text-davinci-003"
        )
        self.DEFAULT_OPENAI_GPT_3_CHAT_MODEL = "gpt-3.5-turbo"
        self.DEFAULT_OPENAI_GPT_4_MODEL = "gpt-4"
        self.DEFAULT_ENCODING = "cl100k_base"
        self.EFAULT_MAX_TOKENS = 2049
        self.TOKEN_OFFSET = 8

        self.MODEL_PREFIXES_TO_MAX_TOKENS = {
            "gpt-4-1106": 128000,
            "gpt-4-32k": 32768,
            "gpt-4": 8192,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-3.5-turbo": 4096,
            "gpt-35-turbo-16k": 16384,
            "gpt-35-turbo": 4096,
            "text-davinci-003": 4097,
            "text-davinci-002": 4097,
            "code-davinci-002": 8001,
            "text-embedding-ada-002": 8191,
            "text-embedding-ada-001": 2046,
        }

        self.EMBEDDING_MODELS = [
            "text-embedding-ada-002",
            "text-embedding-ada-001",
        ]

    @property
    def encoding(self) -> Encoding:
        """
        Returns the encoding for the model.
        If the model is not found, returns the default encoding.
        """
        try:
            return tiktoken.encoding_for_model(self.model)
        except KeyError:
            return tiktoken.get_encoding(self.DEFAULT_ENCODING)

    def default_max_tokens(self) -> int:
        """
        Returns the default maximum number of tokens based on the model.
        """
        tokens = next(
            v
            for k, v in self.MODEL_PREFIXES_TO_MAX_TOKENS.items()
            if self.model.startswith(k)
        )
        offset = (
            0
            if self.model in self.EMBEDDING_MODELS
            else self.TOKEN_OFFSET
        )

        return (
            tokens if tokens else self.DEFAULT_MAX_TOKENS
        ) - offset

    def count_tokens(
        self, text: str | list[dict], model: Optional[str] = None
    ) -> int:
        """
        Counts the number of tokens in the given text.
        If the text is a list of messages, counts the tokens for each message.
        If a model is provided, uses that model for encoding.
        """
        if isinstance(text, list):
            model = model if model else self.model

            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                logging.warning(
                    "model not found. Using cl100k_base encoding."
                )
                encoding = tiktoken.get_encoding("cl100k_base")

            if model in {
                "gpt-3.5-turbo-0613",
                "gpt-3.5-turbo-16k-0613",
                "gpt-4-0314",
                "gpt-4-32k-0314",
                "gpt-4-0613",
                "gpt-4-32k-0613",
            }:
                tokens_per_message = 3
                tokens_per_name = 1
            elif model == "gpt-3.5-turbo-0301":
                tokens_per_message = 4
                tokens_per_name = -1
            elif "gpt-3.5-turbo" in model or "gpt-35-turbo" in model:
                logging.info(
                    "gpt-3.5-turbo may update over time. Returning"
                    " num tokens assuming gpt-3.5-turbo-0613."
                )
                return self.count_tokens(
                    text, model="gpt-3.5-turbo-0613"
                )
            elif "gpt-4" in model:
                logging.info(
                    "gpt-4 may update over time. Returning num tokens"
                    " assuming gpt-4-0613."
                )
                return self.count_tokens(text, model="gpt-4-0613")
            else:
                raise NotImplementedError(
                    "token_count() is not implemented for model"
                    f" {model}. See"
                    " https://github.com/openai/openai-python/blob/main/chatml.md"
                    " for information on how messages are converted"
                    " to tokens."
                )

            num_tokens = 0

            for message in text:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name

            num_tokens += 3

            return num_tokens
        else:
            return len(self.encoding.encode(text))

    def len(self, text: str | list[dict], model: Optional[str]):
        """
        Returns the length of the text in tokens.
        If a model is provided, uses that model for encoding.
        """
        return self.count_tokens(text, model)
