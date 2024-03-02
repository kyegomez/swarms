# HuggingFaceTokenizer Documentation

`HuggingFaceTokenizer` is a comprehensive Python class that leverages the Hugging Face `transformers` library to tokenize text using the SentencePiece tokenization mechanism. This class serves as a convenient wrapper for initializing and using tokenizer models from Hugging Face's transformer models, enabling easy integration of tokenizer functionality in various NLP tasks.

**Purpose and Architecture:**

Tokenization is a critical step in processing natural language wherein text is broken down into smaller elements (tokens), which can be further used for text analysis, language modeling, and other computational linguistics tasks. The `HuggingFaceTokenizer` provides methods to encode text (turning strings into lists of token IDs) and decode lists of token IDs back into human-readable text.

**Table of Contents:**

- [Overview](#overview)
- [Initialization](#initialization)
- [Properties](#properties)
- [Methods](#methods)
- [Usage Examples](#usage-examples)
- [References and Resources](#references-and-resources)

## Overview

The `HuggingFaceTokenizer` class is designed to streamline the process of tokenizing text for natural language processing (NLP). It encapsulates various functionalities, such as encoding text into tokens, decoding tokens into text, and identifying token IDs for special tokens.

## Initialization

`HuggingFaceTokenizer` is initialized by providing the directory containing the pretrained tokenizer model files. During its initialization, it configures its internal state for tokenization processes, prepares access to vocabulary, and establishes necessary properties for subsequent tokenization tasks.

### Constructor Parameters

| Parameter  | Data Type | Description                                | Default |
|------------|-----------|--------------------------------------------|---------|
| model_dir  | `str`     | The directory containing the tokenizer model files. | None    |

### Attributes

| Attribute         | Data Type           | Description                                            |
|-------------------|---------------------|--------------------------------------------------------|
| vocab_size        | `int`               | The size of the vocabulary used by the tokenizer.       |
| bos_token_id      | `int`               | The token ID representing the beginning of sequence token. |
| eos_token_id      | `int`               | The token ID representing the end of sequence token.     |
| prefix_space_tokens | `Set[int]`        | A set of token IDs without a prefix space.                |

## Methods

### Vocabulary Related Methods

#### `vocab_size`
Returns the size of the tokenizer's vocabulary.

#### `bos_token_id`
Returns the token ID used for the beginning of a sentence.

#### `eos_token_id`
Returns the token ID used for the end of a sentence.

#### `prefix_space_tokens`
Returns a set of token IDs that start without prefix spaces.

### Tokenization Methods

#### `encode`
Encodes a given text into a sequence of token IDs.

#### `decode`
Decodes a given sequence of token IDs into human-readable text.

#### `indexes_containing_token`
Returns a list of token IDs that potentially could be decoded into the given token.

#### `__call__`
Tokenizes given text when the object is called like a function.

## Usage Examples

### 1. Initializing the Tokenizer

```python
from swarms.tokenizers import HuggingFaceTokenizer

# Initialize the tokenizer with the path to your tokenizer model.
tokenizer = HuggingFaceTokenizer("/path/to/your/model_dir")
```

### 2. Encoding Text

```python
# Tokenize a single sentence.
sentence = "The quick brown fox jumps over the lazy dog."
token_ids = tokenizer.encode(sentence)
print(token_ids)
```

### 3. Decoding Tokens

```python
# Assuming 'token_ids' contains a list of token IDs
decoded_text = tokenizer.decode(token_ids)
print(decoded_text)
```

### 4. Getting Special Token IDs

```python
# Get the beginning of sequence token ID
bos_id = tokenizer.bos_token_id
print(f"BOS token ID: {bos_id}")

# Get the end of sequence token ID
eos_id = tokenizer.eos_token_id
print(f"EOS token ID: {eos_id}")
```

### 5. Using the Tokenizer

```python
# Tokenize a prompt directly by calling the object with a string.
text = "Hello, world!"
token_ids = tokenizer(text)
print(token_ids)
```

## References and Resources

For more in-depth information on the Hugging Face `transformers` library and SentencePiece, refer to the following resources:

- Hugging Face `transformers` library documentation: https://huggingface.co/docs/transformers/index
- SentencePiece repository and documentation: https://github.com/google/sentencepiece

This documentation provides an introductory overview of the `HuggingFaceTokenizer` class. For a more extensive guide on the various parameters, functionalities, and advanced usage scenarios, users should refer to the detailed library documentation and external resources provided above.
