# Documentation for `swarms.tokenizers.BaseTokenizer` 

## Overview and Introduction

The `swarms.tokenizers` library is designed to provide flexible and efficient tokenization utilities for natural language processing (NLP) tasks. The `BaseTokenizer` class serves as a foundational abstract class from which specific tokenizer implementations can be derived. This class outlines essential functions and properties all tokenizers should have, ensuring consistency and capturing common behaviors required for processing textual data.

## Class Definition: `BaseTokenizer`

### Attributes and Methods

| Name                   | Type                            | Description                                                               |
| ---------------------- | ------------------------------- | ------------------------------------------------------------------------- |
| `max_tokens`           | `int`                           | Maximum number of tokens the tokenizer can process.                       |
| `stop_token`           | `str`                           | Token used to denote the end of processing.                               |
| `stop_sequences`       | `List[str]` (read-only)         | List of stop sequences initialized post-instantiation.                    |
| `count_tokens_left`    | Method: `(text) -> int`         | Computes the number of tokens that can still be added given the text.     |
| `count_tokens`         | Abstract Method: `(text) -> int`| Returns the number of tokens in the given text.                           |

## Functionality and Usage

The `BaseTokenizer` class provides the structure for creating tokenizers. It includes methods for counting the tokens in a given text and determining how many more tokens can be added without exceeding the `max_tokens` limit. This class should be subclassed, and the `count_tokens` method must be implemented in subclasses to provide the specific token counting logic.

### Example: Subclassing `BaseTokenizer`

```python
from swarms.tokenizers import BaseTokenizer


class SimpleTokenizer(BaseTokenizer):

    def count_tokens(self, text: Union[str, List[dict]]) -> int:
        if isinstance(text, str):
            # Split text by spaces as a simple tokenization approach
            return len(text.split())
        elif isinstance(text, list):
            # Assume list of dictionaries with 'token' key
            return sum(len(item["token"].split()) for item in text)
        else:
            raise TypeError("Unsupported type for text")


# Usage example
tokenizer = SimpleTokenizer(max_tokens=100)
text = "This is an example sentence to tokenize."
print(tokenizer.count_tokens(text))  # Outputs: 7 (assuming space tokenization)
remaining_tokens = tokenizer.count_tokens_left(text)
print(remaining_tokens)  # Outputs: 93
```

### Note:

Understand that the `stop_sequences` and `stop_token` in this particular implementation are placeholders to illustrate the pattern. The actual logic may differ based on specific tokenizer requirements.

## Additional Information and Tips

- Tokenization is a vital step in text processing for NLP. It should be tailored to the requirements of the application.
- Ensure that tokenizer definitions are in sync with the models and datasets being used.

## References and Resources

For a deeper understanding of tokenization and its role in NLP, refer to:

- [Natural Language Processing (NLP) in Python — Tokenization](https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html)
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/python/latest/) - a popular library for tokenization, particularly in the context of transformer models.
