# CohereTokenizer Documentation

The `CohereTokenizer` class is designed to interface with Cohere language models and provides methods for tokenizing text inputs. This tokenizer plays a crucial role in preparing data for a Cohere model, which operates on tokens rather than raw text.

---

### Class Name: `CohereTokenizer`

## Overview

The `CohereTokenizer` class is essential for interacting with Cohere models that require tokenized input. As models often operate on tokens, having an intuitive and efficient tokenizer directly linked to the model simplifies preprocessing tasks. This tokenizer counts the tokens in the given text, helping users to manage and understand the tokens they can work with, given limitations like the model's maximum token count.

## Architecture and How the Class Works

The `CohereTokenizer` is built as a data class, ensuring that it is lightweight and focused solely on its data attributes and methods related to tokenization. The class relies on an instance of a Cohere `Client`, which needs to be instantiated with an API key from Cohere before use. 

Upon instantiation, the `CohereTokenizer` holds a reference to a specific Cohere model and interfaces with the `Client` to tokenize text accordingly. It provides a simple utility (`count_tokens`) to count the number of tokens that a string, or a list of strings, would be broken down into by the Cohere API.

## Purpose and Usage

The `CohereTokenizer` is specifically made for users who are working with Cohere language models. It's designed to help them in preprocessing steps by converting text into tokens and determining how many tokens their text segments contain. This is crucial for ensuring that inputs do not exceed the model's maximum token count, as exceeding this limit can result in errors or truncated text.

---

# Class Definition

```python
@dataclass
class CohereTokenizer:
    model: str
    client: Client
    DEFAULT_MODEL: str = "command"
    DEFAULT_MAX_TOKENS: int = 2048
    max_tokens: int = DEFAULT_MAX_TOKENS
```

## Parameters

| Parameter          | Type           | Description                                                   | Default Value |
| ------------------ | -------------- | ------------------------------------------------------------- | ------------- |
| `model`            | `str`          | Specifies the Cohere model to be used for tokenization.       | None          |
| `client`           | `Client`       | An instance of the Cohere client, initialized with an API key.| None          |
| `DEFAULT_MODEL`    | `str`          | The default model to use if none is specified.                | "command"     |
| `DEFAULT_MAX_TOKENS`| `int`         | Default maximum number of tokens the model accepts.           | 2048          |
| `max_tokens`       | `int`          | Maximum number of tokens; it can be altered to fit the model. | `DEFAULT_MAX_TOKENS`|

### Methods

The `CohereTokenizer` class contains the following method:

#### `count_tokens`

```python
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
```

---

# Functionality and Usage Example

Below are examples demonstrating how to use `CohereTokenizer`.

---

## Counting Tokens

### Initialization

First, the Cohere client must be initialized and passed in to create an instance of `CohereTokenizer`.

```python
from cohere import Client

from swarms.tokenizers import CohereTokenizer

# Initialize Cohere client with your API key
cohere_client = Client("your-api-key")

# Instantiate the tokenizer
tokenizer = CohereTokenizer(model="your-model-name", client=cohere_client)
```

### Count Tokens Example 1

Counting tokens for a single string.

```python
text_to_tokenize = "Hello, World!"
token_count = tokenizer.count_tokens(text_to_tokenize)
print(f"Number of tokens: {token_count}")
```

### Count Tokens Example 2

Trying to pass a list instead of a single string, which would raise an error.

```python
texts_to_tokenize = ["Hello, World!", "Another piece of text."]
try:
    token_count = tokenizer.count_tokens(texts_to_tokenize)
except ValueError as e:
    print(f"Error: {e}")
```

The above code would print `Error: Text must be a string.` as the `count_tokens` function expects a string, not a list.

---

# Additional Information and Tips

When working with the `CohereTokenizer`, here are some key points to keep in mind:

- The token count is important to know because Cohere models have a maximum token limit for input. If your text exceeds this limit, it must be split or truncated before being passed to the model.
- It is always a good practice to catch exceptions when using methods like `count_tokens` to handle unexpected inputs gracefully.
- Remember to replace `'your-api-key'` and `'your-model-name'` with your actual Cohere API key and desired model name.

# References and Resources

For more detailed information, refer to the following resources:

- [Cohere API documentation](https://docs.cohere.ai/)
- [Data Classes in Python](https://docs.python.org/3/library/dataclasses.html)

