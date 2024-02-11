# OpenAITokenizer

The `OpenAITokenizer` class is a versatile and intuitive tokenizer designed for use with OpenAI's various language models, including the powerful GPT series. This class addresses the need to efficiently tokenize text for submission to OpenAI's API endpoints, managing different models and their unique tokenization schemes with ease.

Utility of `OpenAITokenizer` centers around its key features:
- Support for multiple OpenAI models including GPT-3 and GPT-4.
- Dynamic token counting that considers model-specific details.
- Straightforward API intended for easy integration with larger systems.

## Architecture and Design

The class adheres to a simple yet effective design, offering methods for calculating token lengths and embedded properties that manage model-specific characteristics such as maximum tokens and encodings. A data class structure is used for clean initializations and better management of class data.

The `OpenAITokenizer` uses a property-based approach and a method-based approach to provide users with a variety of functionalities essential for preparing text input for OpenAI models.

## Attributes

The class contains several key constants and properties that define defaults and settings for use with different models:

| Attribute                                           | Type            | Description                                                 |
|-----------------------------------------------------|-----------------|-------------------------------------------------------------|
| `DEFAULT_OPENAI_GPT_3_COMPLETION_MODEL`             | `str`           | Default completion model for OpenAI GPT-3.                  |
| `DEFAULT_OPENAI_GPT_3_CHAT_MODEL`                   | `str`           | Default chat model for OpenAI GPT-3.                        |
| `DEFAULT_OPENAI_GPT_4_MODEL`                        | `str`           | Default model for OpenAI GPT-4.                             |
| `DEFAULT_ENCODING`                                  | `str`           | Default encoding for text.                                  |
| `DEFAULT_MAX_TOKENS`                                | `int`           | Default maximum number of tokens based on the model.        |
| `TOKEN_OFFSET`                                      | `int`           | Token offset applicable to some models.                     |
| `MODEL_PREFIXES_TO_MAX_TOKENS`                      | `dict`          | Mapping of model prefixes to their respective max tokens.   |
| `EMBEDDING_MODELS`                                  | `list`          | List of embedding models supported.                         |
| `model`                                             | `str`           | Name of the model currently being used.                     |

## Methods

The `OpenAITokenizer` class offers a variety of methods:

| Method                | Arguments                                   | Return Type | Description                                                                                    |
|-----------------------|---------------------------------------------|-------------|------------------------------------------------------------------------------------------------|
| `__post_init__`       | None                                        | `None`      | Method called after the class has been initialized to set up default values.                  |
| `encoding`            | None                                        | `Encoding`  | Getter method that retrieves the encoding based on the specified model.                        |
| `default_max_tokens`  | None                                        | `int`       | Calculates the default max tokens based on the current model or defaults if not model-specific.|
| `count_tokens`        | `text: str \| list[dict]`, `model: str`    | `int`       | Counts the number of tokens within a given text or a list of messages.                         |
| `len`                 | `text: str \| list[dict]`, `model: str`    | `int`       | Wrapper for `count_tokens`, providing a more intuitive naming convention.                      |

### Usage Examples

Given the extensive nature of this class, several examples are provided for each method, detailing how to use the `OpenAITokenizer` in different contexts.

#### Example 1: Initializing the Tokenizer

```python
from swarms.tokenizers import OpenAITokenizer

tokenizer = OpenAITokenizer(model='gpt-4')
```

This example creates a new instance of `OpenAITokenizer` set to work with the GPT-4 model.

#### Example 2: Counting Tokens

```python
text = "Hello, this is an example text to tokenize."

# Initialize the tokenizer
tokenizer = OpenAITokenizer(model='gpt-4')

# Count tokens
num_tokens = tokenizer.count_tokens(text)
print(f"Number of tokens: {num_tokens}")
```

This code snippet demonstrates how to count the number of tokens in a string of text using the specified model's encoding.

#### Example 3: Custom Model Token Counting

```python
messages = [
    {"name": "Alice", "message": "Hello! How are you?"},
    {"name": "Bob", "message": "I'm good! Just working on some code."},
]

tokenizer = OpenAITokenizer(model='gpt-3.5-turbo')

# Count tokens for a list of messages
num_tokens = tokenizer.len(messages, model="gpt-3.5-turbo-0613")
print(f"Total tokens for messages: {num_tokens}")
```

In this example, we're invoking the `len` method to count the tokens in a conversation thread. Each message is represented as a dictionary with a `name` and `message` field.

