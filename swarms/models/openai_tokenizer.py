from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import tiktoken
from attr import Factory, define, field


@define(frozen=True)
class BaseTokenizer(ABC):
    DEFAULT_STOP_SEQUENCES = ["Observation:"]

    stop_sequences: list[str] = field(
        default=Factory(lambda: BaseTokenizer.DEFAULT_STOP_SEQUENCES),
        kw_only=True,
    )

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        ...

    def count_tokens_left(self, text: str) -> int:
        diff = self.max_tokens - self.count_tokens(text)

        if diff > 0:
            return diff
        else:
            return 0

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        ...


@define(frozen=True)
class OpenAITokenizer(BaseTokenizer):
    DEFAULT_OPENAI_GPT_3_COMPLETION_MODEL = "text-davinci-003"
    DEFAULT_OPENAI_GPT_3_CHAT_MODEL = "gpt-3.5-turbo"
    DEFAULT_OPENAI_GPT_4_MODEL = "gpt-4"
    DEFAULT_ENCODING = "cl100k_base"
    DEFAULT_MAX_TOKENS = 2049
    TOKEN_OFFSET = 8

    MODEL_PREFIXES_TO_MAX_TOKENS = {
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

    EMBEDDING_MODELS = ["text-embedding-ada-002", "text-embedding-ada-001"]

    model: str = field(kw_only=True)

    @property
    def encoding(self) -> tiktoken.Encoding:
        try:
            return tiktoken.encoding_for_model(self.model)
        except KeyError:
            return tiktoken.get_encoding(self.DEFAULT_ENCODING)

    @property
    def max_tokens(self) -> int:
        tokens = next(v for k, v in self.MODEL_PREFIXES_TO_MAX_TOKENS.items()
                      if self.model.startswith(k))
        offset = 0 if self.model in self.EMBEDDING_MODELS else self.TOKEN_OFFSET

        return (tokens if tokens else self.DEFAULT_MAX_TOKENS) - offset

    def count_tokens(self,
                     text: str | list,
                     model: Optional[str] = None) -> int:
        """
        Handles the special case of ChatML. Implementation adopted from the official OpenAI notebook:
        https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """
        if isinstance(text, list):
            model = model if model else self.model

            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                logging.warning("model not found. Using cl100k_base encoding.")

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
                # every message follows <|start|>{role/name}\n{content}<|end|>\n
                tokens_per_message = 4
                # if there's a name, the role is omitted
                tokens_per_name = -1
            elif "gpt-3.5-turbo" in model or "gpt-35-turbo" in model:
                logging.info(
                    "gpt-3.5-turbo may update over time. Returning num tokens assuming"
                    " gpt-3.5-turbo-0613.")
                return self.count_tokens(text, model="gpt-3.5-turbo-0613")
            elif "gpt-4" in model:
                logging.info(
                    "gpt-4 may update over time. Returning num tokens assuming"
                    " gpt-4-0613.")
                return self.count_tokens(text, model="gpt-4-0613")
            else:
                raise NotImplementedError(
                    f"""token_count() is not implemented for model {model}.
                    See https://github.com/openai/openai-python/blob/main/chatml.md for
                    information on how messages are converted to tokens.""")

            num_tokens = 0

            for message in text:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name

            # every reply is primed with <|start|>assistant<|message|>
            num_tokens += 3

            return num_tokens
        else:
            return len(
                self.encoding.encode(text,
                                     allowed_special=set(self.stop_sequences)))
