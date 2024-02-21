# PdfChunker Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Overview](#overview)
3. [Installation](#installation)
4. [Usage](#usage)
   1. [PdfChunker Class](#pdfchunker-class)
   2. [Examples](#examples)
5. [Additional Information](#additional-information)
6. [Conclusion](#conclusion)

---

## 1. Introduction <a name="introduction"></a>

The `PdfChunker` module is a specialized tool designed to split PDF text content into smaller, more manageable chunks. It is a valuable asset for processing PDF documents in natural language processing and text analysis tasks.

This documentation provides a comprehensive guide on how to use the `PdfChunker` module. It covers its purpose, parameters, and usage, ensuring that you can effectively process PDF text content.

---

## 2. Overview <a name="overview"></a>

The `PdfChunker` module serves a critical role in handling PDF text content, which is often lengthy and complex. Key features and parameters of the `PdfChunker` module include:

- `separators`: Specifies a list of `ChunkSeparator` objects used to split the PDF text content into chunks.
- `tokenizer`: Defines the tokenizer used for counting tokens in the text.
- `max_tokens`: Sets the maximum token limit for each chunk.

By using the `PdfChunker`, you can efficiently prepare PDF text content for further analysis and processing.

---

## 3. Installation <a name="installation"></a>

Before using the `PdfChunker` module, ensure you have the required dependencies installed. The module relies on the `swarms` library. You can install this dependency using pip:

```bash
pip install swarms
```

---

## 4. Usage <a name="usage"></a>

In this section, we'll explore how to use the `PdfChunker` module effectively. It consists of the `PdfChunker` class and provides examples to demonstrate its usage.

### 4.1. `PdfChunker` Class <a name="pdfchunker-class"></a>

The `PdfChunker` class is the core component of the `PdfChunker` module. It is used to create a `PdfChunker` instance, which can split PDF text content into manageable chunks.

#### Parameters:
- `separators` (list[ChunkSeparator]): Specifies a list of `ChunkSeparator` objects used to split the PDF text content into chunks.
- `tokenizer` (OpenAITokenizer): Defines the tokenizer used for counting tokens in the text.
- `max_tokens` (int): Sets the maximum token limit for each chunk.

### 4.2. Examples <a name="examples"></a>

Let's explore how to use the `PdfChunker` class with different scenarios and applications.

#### Example 1: Basic Chunking

```python
from swarms.chunkers.chunk_seperator import ChunkSeparator
from swarms.chunkers.pdf_chunker import PdfChunker

# Initialize the PdfChunker
pdf_chunker = PdfChunker()

# PDF text content to be chunked
pdf_text = "This is a PDF document with multiple paragraphs and sentences. It should be split into smaller chunks for analysis."

# Chunk the PDF text content
chunks = pdf_chunker.chunk(pdf_text)

# Print the generated chunks
for idx, chunk in enumerate(chunks, start=1):
    print(f"Chunk {idx}:\n{chunk.value}")
```

#### Example 2: Custom Separators

```python
from swarms.chunkers.chunk_seperator import ChunkSeparator
from swarms.chunkers.pdf_chunker import PdfChunker

# Define custom separators for PDF chunking
custom_separators = [ChunkSeparator("\n\n"), ChunkSeparator(". ")]

# Initialize the PdfChunker with custom separators
pdf_chunker = PdfChunker(separators=custom_separators)

# PDF text content with custom separators
pdf_text = "This PDF document has custom paragraph separators.\n\nIt also uses period-based sentence separators. Split accordingly."

# Chunk the PDF text content
chunks = pdf_chunker.chunk(pdf_text)

# Print the generated chunks
for idx, chunk in enumerate(chunks, start=1):
    print(f"Chunk {idx}:\n{chunk.value}")
```

#### Example 3: Adjusting Maximum Tokens

```python
from swarms.chunkers.pdf_chunker import PdfChunker

# Initialize the PdfChunker with a custom maximum token limit
pdf_chunker = PdfChunker(max_tokens=50)

# Lengthy PDF text content
pdf_text = "This is an exceptionally long PDF document that should be broken into smaller chunks based on token count."

# Chunk the PDF text content
chunks = pdf_chunker.chunk(pdf_text)

# Print the generated chunks
for idx, chunk in enumerate(chunks, start=1):
    print(f"Chunk {idx}:\n{chunk.value}")
```

### 4.3. Additional Features

The `PdfChunker` class also provides additional features:

#### Recursive Chunking
The `_chunk_recursively` method handles the recursive chunking of PDF text content, ensuring that each chunk stays within the token limit.

---

## 5. Additional Information <a name="additional-information"></a>

- **PDF Text Chunking**: The `PdfChunker` module is a specialized tool for splitting PDF text content into manageable chunks, making it suitable for natural language processing tasks involving PDF documents.
- **Custom Separators**: You can customize separators to adapt the PDF text content chunking process to specific document structures.
- **Token Count**: The module accurately counts tokens using the specified tokenizer, ensuring that chunks do not exceed token limits.

---

## 6. Conclusion <a name="conclusion"></a>

The `PdfChunker` module is a valuable asset for processing PDF text content in various natural language processing and text analysis tasks. This documentation has provided a comprehensive guide on its usage, parameters, and examples, ensuring that you can effectively prepare PDF documents for further analysis and processing.

By using the `PdfChunker`, you can efficiently break down lengthy and complex PDF text content into manageable chunks, making it ready for in-depth analysis.

*Please check the official `PdfChunker` repository and documentation for any updates beyond the knowledge cutoff date.*