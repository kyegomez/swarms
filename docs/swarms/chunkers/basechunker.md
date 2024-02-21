# BaseChunker Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Overview](#overview)
3. [Installation](#installation)
4. [Usage](#usage)
   1. [BaseChunker Class](#basechunker-class)
   2. [Examples](#examples)
5. [Additional Information](#additional-information)
6. [Conclusion](#conclusion)

---

## 1. Introduction <a name="introduction"></a>

The `BaseChunker` module is a tool for splitting text into smaller chunks that can be processed by a language model. It is a fundamental component in natural language processing tasks that require handling long or complex text inputs.

This documentation provides an extensive guide on using the `BaseChunker` module, explaining its purpose, parameters, and usage.

---

## 2. Overview <a name="overview"></a>

The `BaseChunker` module is designed to address the challenge of processing lengthy text inputs that exceed the maximum token limit of language models. By breaking such text into smaller, manageable chunks, it enables efficient and accurate processing.

Key features and parameters of the `BaseChunker` module include:
- `separators`: Specifies a list of `ChunkSeparator` objects used to split the text into chunks.
- `tokenizer`: Defines the tokenizer to be used for counting tokens in the text.
- `max_tokens`: Sets the maximum token limit for each chunk.

The `BaseChunker` module facilitates the chunking process and ensures that the generated chunks are within the token limit.

---

## 3. Installation <a name="installation"></a>

Before using the `BaseChunker` module, ensure you have the required dependencies installed. The module relies on `griptape` and `swarms` libraries. You can install these dependencies using pip:

```bash
pip install griptape swarms
```

---

## 4. Usage <a name="usage"></a>

In this section, we'll cover how to use the `BaseChunker` module effectively. It consists of the `BaseChunker` class and provides examples to demonstrate its usage.

### 4.1. `BaseChunker` Class <a name="basechunker-class"></a>

The `BaseChunker` class is the core component of the `BaseChunker` module. It is used to create a `BaseChunker` instance, which can split text into chunks efficiently.

#### Parameters:
- `separators` (list[ChunkSeparator]): Specifies a list of `ChunkSeparator` objects used to split the text into chunks.
- `tokenizer` (OpenAITokenizer): Defines the tokenizer to be used for counting tokens in the text.
- `max_tokens` (int): Sets the maximum token limit for each chunk.

### 4.2. Examples <a name="examples"></a>

Let's explore how to use the `BaseChunker` class with different scenarios and applications.

#### Example 1: Basic Chunking

```python
from basechunker import BaseChunker, ChunkSeparator

# Initialize the BaseChunker
chunker = BaseChunker()

# Text to be chunked
input_text = (
    "This is a long text that needs to be split into smaller chunks for processing."
)

# Chunk the text
chunks = chunker.chunk(input_text)

# Print the generated chunks
for idx, chunk in enumerate(chunks, start=1):
    print(f"Chunk {idx}: {chunk.value}")
```

#### Example 2: Custom Separators

```python
from basechunker import BaseChunker, ChunkSeparator

# Define custom separators
custom_separators = [ChunkSeparator(","), ChunkSeparator(";")]

# Initialize the BaseChunker with custom separators
chunker = BaseChunker(separators=custom_separators)

# Text with custom separators
input_text = "This text, separated by commas; should be split accordingly."

# Chunk the text
chunks = chunker.chunk(input_text)

# Print the generated chunks
for idx, chunk in enumerate(chunks, start=1):
    print(f"Chunk {idx}: {chunk.value}")
```

#### Example 3: Adjusting Maximum Tokens

```python
from basechunker import BaseChunker

# Initialize the BaseChunker with a custom maximum token limit
chunker = BaseChunker(max_tokens=50)

# Long text input
input_text = "This is an exceptionally long text that should be broken into smaller chunks based on token count."

# Chunk the text
chunks = chunker.chunk(input_text)

# Print the generated chunks
for idx, chunk in enumerate(chunks, start=1):
    print(f"Chunk {idx}: {chunk.value}")
```

### 4.3. Additional Features

The `BaseChunker` class also provides additional features:

#### Recursive Chunking
The `_chunk_recursively` method handles the recursive chunking of text, ensuring that each chunk stays within the token limit.

---

## 5. Additional Information <a name="additional-information"></a>

- **Text Chunking**: The `BaseChunker` module is a fundamental tool for text chunking, a crucial step in preprocessing text data for various natural language processing tasks.
- **Custom Separators**: You can customize the separators used to split the text, allowing flexibility in how text is chunked.
- **Token Count**: The module accurately counts tokens using the specified tokenizer, ensuring that chunks do not exceed token limits.

---

## 6. Conclusion <a name="conclusion"></a>

The `BaseChunker` module is an essential tool for text preprocessing and handling long or complex text inputs in natural language processing tasks. This documentation has provided a comprehensive guide on its usage, parameters, and examples, enabling you to efficiently manage and process text data by splitting it into manageable chunks.

By using the `BaseChunker`, you can ensure that your text data remains within token limits and is ready for further analysis and processing.

*Please check the official `BaseChunker` repository and documentation for any updates beyond the knowledge cutoff date.*