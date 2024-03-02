# swarms.tokenizers Documentation

`swarms.tokenizers` is a PyTorch-like tokenization library designed to facilitate natural language processing (NLP) tasks by converting text inputs into a form that machine learning models can interpret. In this documentation, we will outline how to utilize the `SentencePieceTokenizer` class from the `swarms.tokenizers` library, which offers sentencepiece tokenization, a language-independent subword tokenizer and detokenizer.

## Purpose and Architecture of `SentencePieceTokenizer`

The `SentencePieceTokenizer` class uses a pre-trained sentencepiece model to tokenize and detokenize texts. SentencePiece is an unsupervised text tokenizer and detokenizer that allows the generation of a subword vocabulary from raw data. By breaking text down into subword units (like wordpieces or byte-pair-encodings), SentencePiece handles languages without a clear word boundary and can improve the performance of text processing in neural network models.

In `SentencePieceTokenizer`, the tokenization process is language-agnostic and encompasses a range of tokenization strategies, such as byte pair encoding (BPE), unigram, or a combination of both. The class is designed with ease of use in mind, allowing seamless integration with other components of the NLP pipeline.

## Class Definition

```python
class SentencePieceTokenizer:
    """
    Tokenizer of sentencepiece.

    Args:
        model_file (str): the path of the tokenizer model
    """
```

## Initialization Parameters

Property/Method | Type | Description
----------------|------|-------------
`model_file` | `str` | The path to the pretrained sentencepiece model file.

## Methods and Usage

Below, we detail the methods available in `SentencePieceTokenizer`, including their parameters, their functionality, and usage examples.

### Method: `__init__`

Instantiates an instance of the `SentencePieceTokenizer` with the specified sentencepiece model.

#### Parameters

Parameter | Type | Description
----------|------|-------------
`model_file` | `str` | The path to the pretrained sentencepiece model file.

#### Example

```python
from swarms.tokenizers import SentencePieceTokenizer

tokenizer = SentencePieceTokenizer(model_file="your_model.model")
```

### Properties: Vocabulary Information

These properties provide access to various vocabulary-specific information.

#### `vocab_size`
#### `bos_token_id`
#### `eos_token_id`

##### Example

```python
vocab_size = tokenizer.vocab_size
print(f"Vocabulary size: {vocab_size}")

bos_id = tokenizer.bos_token_id
eos_id = tokenizer.eos_token_id
print(f"BOS token ID: {bos_id}, EOS token ID: {eos_id}")
```

### Method: `indexes_containing_token`

Finds possible tokenizer indexes that, when decoded, may contain the input token.

#### Parameters

Parameter | Type | Description
----------|------|-------------
`token` | `str` | The token for which possible indexes are to be found.

#### Returns

- `List[int]`: List of tokenizer indexes that might contain the token.

#### Example

```python
indexes = tokenizer.indexes_containing_token("▁the")
print(f"Indexes containing '▁the': {indexes}")
```

### Method: `encode`

Tokenizes a text prompt into a list of token IDs.

#### Parameters

Parameter | Type | Description
----------|------|-------------
`s` | `str` | The text prompt to tokenize.
`add_bos` | `bool` | If `True`, it adds the beginning-of-sentence token. (default: `True`)

#### Returns
- `List[int]`: List of token IDs representing the text prompt.

#### Example

```python
encoded_ids = tokenizer.encode("Hello, world!", add_bos=True)
print(f"Encoded token IDs: {encoded_ids}")
```

### Method: `decode`

Detokenizes a list of token IDs into text.

#### Parameters

Parameter | Type | Description
----------|------|-------------
`t` | `List[int]` | A list of token IDs to detokenize.
`offset` | `Optional[int]` | For incremental decoding. Defaults to `None`, which means it is not applied.

#### Returns

- `str`: Text representation of the decoded token IDs.

#### Example

```python
decoded_text = tokenizer.decode([bos_id] + encoded_ids)
print(f"Decoded text: {decoded_text}")
```

### Method: `__call__`

Tokenizes prompts when the class instance is used as a callable.

#### Parameters

Parameter | Type | Description
----------|------|-------------
`s` | `Union[str, Sequence[str]]` | Text prompts to tokenize.
`add_bos` | `bool` | If `True`, it adds the beginning-of-sentence token. (default: `False`)
`add_eos` | `bool` | If `True`, it adds the end-of-sentence token. (default: `False`)

#### Returns

- `addict.Addict`: Object with `input_ids` containing the list of token IDs.

#### Example

```python
input_data = tokenizer("Let's tokenize this sentence.")
print(f"Tokenized input IDs: {input_data.input_ids}")
```

## Additional Information and Tips

The library has efficient internals that cache information for performance benefits. For example, `indexes_containing_token` uses a deque to store the most recent lookups, which saves computation time by avoiding re-traversing the vocabulary.

## Conclusion

This documentation provides an in-depth explanation of `swarms.tokenizers` with a focus on the `SentencePieceTokenizer` class. By following the examples and guidance detailed above, users should be able to effectively use the tokenizers for their NLP tasks. Users are also encouraged to refer to further resources and the official SentencePiece documentation for more advanced use cases and configurations.
