"""Contains classes for querying large language models."""
from math import ceil
import os
import time
from tqdm import tqdm
from abc import ABC, abstractmethod

import openai

gpt_costs_per_thousand = {
    "davinci": 0.0200,
    "curie": 0.0020,
    "babbage": 0.0005,
    "ada": 0.0004,
}


def model_from_config(config, disable_tqdm=True):
    """Returns a model based on the config."""
    model_type = config["name"]
    if model_type == "GPT_forward":
        return GPT_Forward(config, disable_tqdm=disable_tqdm)
    elif model_type == "GPT_insert":
        return GPT_Insert(config, disable_tqdm=disable_tqdm)
    raise ValueError(f"Unknown model type: {model_type}")


class LLM(ABC):
    """Abstract base class for large language models."""

    @abstractmethod
    def generate_text(self, prompt):
        """Generates text from the model.
        Parameters:
            prompt: The prompt to use. This can be a string or a list of strings.
        Returns:
            A list of strings.
        """
        pass

    @abstractmethod
    def log_probs(self, text, log_prob_range):
        """Returns the log probs of the text.
        Parameters:
            text: The text to get the log probs of. This can be a string or a list of strings.
            log_prob_range: The range of characters within each string to get the log_probs of.
                This is a list of tuples of the form (start, end).
        Returns:
            A list of log probs.
        """
        pass


class GPT_Forward(LLM):
    """Wrapper for GPT-3."""

    def __init__(self, config, needs_confirmation=False, disable_tqdm=True):
        """Initializes the model."""
        self.config = config
        self.needs_confirmation = needs_confirmation
        self.disable_tqdm = disable_tqdm

    def confirm_cost(self, texts, n, max_tokens):
        total_estimated_cost = 0
        for text in texts:
            total_estimated_cost += (
                gpt_get_estimated_cost(self.config, text, max_tokens) * n
            )
        print(f"Estimated cost: ${total_estimated_cost:.2f}")
        # Ask the user to confirm in the command line
        if os.getenv("LLM_SKIP_CONFIRM") is None:
            confirm = input("Continue? (y/n) ")
            if confirm != "y":
                raise Exception("Aborted.")

    def auto_reduce_n(self, fn, prompt, n):
        """Reduces n by half until the function succeeds."""
        try:
            return fn(prompt, n)
        except BatchSizeException as e:
            if n == 1:
                raise e
            return self.auto_reduce_n(fn, prompt, n // 2) + self.auto_reduce_n(
                fn, prompt, n // 2
            )

    def generate_text(self, prompt, n):
        if not isinstance(prompt, list):
            prompt = [prompt]
        if self.needs_confirmation:
            self.confirm_cost(prompt, n, self.config["gpt_config"]["max_tokens"])
        batch_size = self.config["batch_size"]
        prompt_batches = [
            prompt[i : i + batch_size] for i in range(0, len(prompt), batch_size)
        ]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Generating {len(prompt) * n} completions, "
                f"split into {len(prompt_batches)} batches of size {batch_size * n}"
            )
        text = []

        for prompt_batch in tqdm(prompt_batches, disable=self.disable_tqdm):
            text += self.auto_reduce_n(self.__generate_text, prompt_batch, n)
        return text

    def complete(self, prompt, n):
        """Generates text from the model and returns the log prob data."""
        if not isinstance(prompt, list):
            prompt = [prompt]
        batch_size = self.config["batch_size"]
        prompt_batches = [
            prompt[i : i + batch_size] for i in range(0, len(prompt), batch_size)
        ]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Generating {len(prompt) * n} completions, "
                f"split into {len(prompt_batches)} batches of size {batch_size * n}"
            )
        res = []
        for prompt_batch in tqdm(prompt_batches, disable=self.disable_tqdm):
            res += self.__complete(prompt_batch, n)
        return res

    def log_probs(self, text, log_prob_range=None):
        """Returns the log probs of the text."""
        if not isinstance(text, list):
            text = [text]
        if self.needs_confirmation:
            self.confirm_cost(text, 1, 0)
        batch_size = self.config["batch_size"]
        text_batches = [
            text[i : i + batch_size] for i in range(0, len(text), batch_size)
        ]
        if log_prob_range is None:
            log_prob_range_batches = [None] * len(text)
        else:
            assert len(log_prob_range) == len(text)
            log_prob_range_batches = [
                log_prob_range[i : i + batch_size]
                for i in range(0, len(log_prob_range), batch_size)
            ]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Getting log probs for {len(text)} strings, "
                f"split into {len(text_batches)} batches of (maximum) size {batch_size}"
            )
        log_probs = []
        tokens = []
        for text_batch, log_prob_range in tqdm(
            list(zip(text_batches, log_prob_range_batches)), disable=self.disable_tqdm
        ):
            log_probs_batch, tokens_batch = self.__log_probs(text_batch, log_prob_range)
            log_probs += log_probs_batch
            tokens += tokens_batch
        return log_probs, tokens

    def __generate_text(self, prompt, n):
        """Generates text from the model."""
        if not isinstance(prompt, list):
            text = [prompt]
        config = self.config["gpt_config"].copy()
        config["n"] = n
        # If there are any [APE] tokens in the prompts, remove them
        for i in range(len(prompt)):
            prompt[i] = prompt[i].replace("[APE]", "").strip()
        response = None
        while response is None:
            try:
                response = openai.Completion.create(**config, prompt=prompt)
            except Exception as e:
                if "is greater than the maximum" in str(e):
                    raise BatchSizeException()
                print(e)
                print("Retrying...")
                time.sleep(5)

        return [response["choices"][i]["text"] for i in range(len(response["choices"]))]

    def __complete(self, prompt, n):
        """Generates text from the model and returns the log prob data."""
        if not isinstance(prompt, list):
            text = [prompt]
        config = self.config["gpt_config"].copy()
        config["n"] = n
        # If there are any [APE] tokens in the prompts, remove them
        for i in range(len(prompt)):
            prompt[i] = prompt[i].replace("[APE]", "").strip()
        response = None
        while response is None:
            try:
                response = openai.Completion.create(**config, prompt=prompt)
            except Exception as e:
                print(e)
                print("Retrying...")
                time.sleep(5)
        return response["choices"]

    def __log_probs(self, text, log_prob_range=None):
        """Returns the log probs of the text."""
        if not isinstance(text, list):
            text = [text]
        if log_prob_range is not None:
            for i in range(len(text)):
                lower_index, upper_index = log_prob_range[i]
                assert lower_index < upper_index
                assert lower_index >= 0
                assert upper_index - 1 < len(text[i])
        config = self.config["gpt_config"].copy()
        config["logprobs"] = 1
        config["echo"] = True
        config["max_tokens"] = 0
        if isinstance(text, list):
            text = [f"\n{text[i]}" for i in range(len(text))]
        else:
            text = f"\n{text}"
        response = None
        while response is None:
            try:
                response = openai.Completion.create(**config, prompt=text)
            except Exception as e:
                print(e)
                print("Retrying...")
                time.sleep(5)
        log_probs = [
            response["choices"][i]["logprobs"]["token_logprobs"][1:]
            for i in range(len(response["choices"]))
        ]
        tokens = [
            response["choices"][i]["logprobs"]["tokens"][1:]
            for i in range(len(response["choices"]))
        ]
        offsets = [
            response["choices"][i]["logprobs"]["text_offset"][1:]
            for i in range(len(response["choices"]))
        ]

        # Subtract 1 from the offsets to account for the newline
        for i in range(len(offsets)):
            offsets[i] = [offset - 1 for offset in offsets[i]]

        if log_prob_range is not None:
            # First, we need to find the indices of the tokens in the log probs
            # that correspond to the tokens in the log_prob_range
            for i in range(len(log_probs)):
                lower_index, upper_index = self.get_token_indices(
                    offsets[i], log_prob_range[i]
                )
                log_probs[i] = log_probs[i][lower_index:upper_index]
                tokens[i] = tokens[i][lower_index:upper_index]

        return log_probs, tokens

    def get_token_indices(self, offsets, log_prob_range):
        """Returns the indices of the tokens in the log probs that correspond to the tokens in the log_prob_range."""
        # For the lower index, find the highest index that is less than or equal to the lower index
        lower_index = 0
        for i in range(len(offsets)):
            if offsets[i] <= log_prob_range[0]:
                lower_index = i
            else:
                break

        upper_index = len(offsets)
        for i in range(len(offsets)):
            if offsets[i] >= log_prob_range[1]:
                upper_index = i
                break

        return lower_index, upper_index


class GPT_Insert(LLM):
    def __init__(self, config, needs_confirmation=False, disable_tqdm=True):
        """Initializes the model."""
        self.config = config
        self.needs_confirmation = needs_confirmation
        self.disable_tqdm = disable_tqdm

    def confirm_cost(self, texts, n, max_tokens):
        total_estimated_cost = 0
        for text in texts:
            total_estimated_cost += (
                gpt_get_estimated_cost(self.config, text, max_tokens) * n
            )
        print(f"Estimated cost: ${total_estimated_cost:.2f}")
        # Ask the user to confirm in the command line
        if os.getenv("LLM_SKIP_CONFIRM") is None:
            confirm = input("Continue? (y/n) ")
            if confirm != "y":
                raise Exception("Aborted.")

    def auto_reduce_n(self, fn, prompt, n):
        """Reduces n by half until the function succeeds."""
        try:
            return fn(prompt, n)
        except BatchSizeException as e:
            if n == 1:
                raise e
            return self.auto_reduce_n(fn, prompt, n // 2) + self.auto_reduce_n(
                fn, prompt, n // 2
            )

    def generate_text(self, prompt, n):
        if not isinstance(prompt, list):
            prompt = [prompt]
        if self.needs_confirmation:
            self.confirm_cost(prompt, n, self.config["gpt_config"]["max_tokens"])
        batch_size = self.config["batch_size"]
        assert batch_size == 1
        prompt_batches = [
            prompt[i : i + batch_size] for i in range(0, len(prompt), batch_size)
        ]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Generating {len(prompt) * n} completions, split into {len(prompt_batches)} batches of (maximum) size {batch_size * n}"
            )
        text = []
        for prompt_batch in tqdm(prompt_batches, disable=self.disable_tqdm):
            text += self.auto_reduce_n(self.__generate_text, prompt_batch, n)
        return text

    def log_probs(self, text, log_prob_range=None):
        raise NotImplementedError

    def __generate_text(self, prompt, n):
        """Generates text from the model."""
        config = self.config["gpt_config"].copy()
        config["n"] = n
        # Split prompts into prefixes and suffixes with the [APE] token (do not include the [APE] token in the suffix)
        prefix = prompt[0].split("[APE]")[0]
        suffix = prompt[0].split("[APE]")[1]
        response = None
        while response is None:
            try:
                response = openai.Completion.create(
                    **config, prompt=prefix, suffix=suffix
                )
            except Exception as e:
                print(e)
                print("Retrying...")
                time.sleep(5)

        # Remove suffix from the generated text
        texts = [
            response["choices"][i]["text"].replace(suffix, "")
            for i in range(len(response["choices"]))
        ]
        return texts


def gpt_get_estimated_cost(config, prompt, max_tokens):
    """Uses the current API costs/1000 tokens to estimate the cost of generating text from the model."""
    # Get rid of [APE] token
    prompt = prompt.replace("[APE]", "")
    # Get the number of tokens in the prompt
    n_prompt_tokens = len(prompt) // 4
    # Get the number of tokens in the generated text
    total_tokens = n_prompt_tokens + max_tokens
    engine = config["gpt_config"]["model"].split("-")[1]
    costs_per_thousand = gpt_costs_per_thousand
    if engine not in costs_per_thousand:
        # Try as if it is a fine-tuned model
        engine = config["gpt_config"]["model"].split(":")[0]
        costs_per_thousand = {
            "davinci": 0.1200,
            "curie": 0.0120,
            "babbage": 0.0024,
            "ada": 0.0016,
        }
    price = costs_per_thousand[engine] * total_tokens / 1000
    return price


class BatchSizeException(Exception):
    pass
