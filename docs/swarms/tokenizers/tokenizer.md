# `Tokenizer` Class Documentation

The `Tokenizer` class is a flexible and robust tokenization tool designed to efficiently tokenize prompts into a sequence of token IDs or convert token IDs back into readable text. The class works by initializing with a path to a pretrained tokenization model and supports different tokenization backends based on the availability of configs and pretrained models.

## Initialization & Configuration

### Parameters:

| Parameter  | Type | Description                              | Required |
|------------|------|------------------------------------------|----------|
| model_file | str  | Path to the tokenizer model or directory | Yes      |

### Attributes:

| Attribute        | Type | Description                        |
|------------------|------|------------------------------------|
| vocab_size       | int  | Size of the tokenizer's vocabulary |
| bos_token_id     | int  | ID of the beginning-of-sequence token |
| eos_token_id     | int  | ID of the end-of-sequence token    |

### Methods:

| Method                         | Returns | Description                                                  |
|--------------------------------|---------|--------------------------------------------------------------|
| encode(s, add_bos=True, **kwargs) | list[int] | Tokenizes a prompt and returns token IDs.                   |
| decode(t, offset=None)        | str     | Decodes a list of token IDs to a string.                     |
| __call__(s)                   | list[int] | Tokenize prompts when the instance is called directly.       |
| indexes_containing_token(token) | list[int] | Returns indexes in the vocabulary that may contain the token. |

## Usage Examples

### Tokenizing a Prompt

```python
from swarms.tokenizers import Tokenizer

tokenizer = Tokenizer("/path/to/tokenizer.model")

# Tokenize a single prompt string
prompt = "Hello, world!"
token_ids = tokenizer.encode(prompt)
print(token_ids)
```

### Decoding Token IDs

```python
# Decode token IDs back into text
decoded_text = tokenizer.decode(token_ids)
print(decoded_text)
```

### Incremental Decoding

```python
# Incremental decoding with offset (useful for streaming applications)
partial_tokens = [token_ids[0]]  # simulate partially received tokens
decoded_partial = tokenizer.decode(partial_tokens, offset=0)
print(decoded_partial)
```

### Properties Access

```python
# Access vocabulary size and special token IDs
print("Vocabulary Size:", tokenizer.vocab_size)
print("BOS Token ID:", tokenizer.bos_token_id)
print("EOS Token ID:", tokenizer.eos_token_id)
```

### Indexes Containing Token

```python
# Find indexes that may output a specific token during decoding
token = "world"
indexes = tokenizer.indexes_containing_token(token)
print("Token Indexes:", indexes)
```
