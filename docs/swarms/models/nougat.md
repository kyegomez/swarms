# `Nougat` Documentation

## Introduction

Welcome to the documentation for Nougat, a versatile model designed by Meta for transcribing scientific PDFs into user-friendly Markdown format, extracting information from PDFs, and extracting metadata from PDF documents. This documentation will provide you with a deep understanding of the Nougat class, its architecture, usage, and examples.

## Overview

Nougat is a powerful tool that combines language modeling and image processing capabilities to convert scientific PDF documents into Markdown format. It is particularly useful for researchers, students, and professionals who need to extract valuable information from PDFs quickly. With Nougat, you can simplify complex PDFs, making their content more accessible and easy to work with.

## Class Definition

```python
class Nougat:
    def __init__(
        self,
        model_name_or_path="facebook/nougat-base",
        min_length: int = 1,
        max_new_tokens: int = 30,
    ):
```

## Purpose

The Nougat class serves the following primary purposes:

1. **PDF Transcription**: Nougat is designed to transcribe scientific PDFs into Markdown format. It helps convert complex PDF documents into a more readable and structured format, making it easier to extract information.

2. **Information Extraction**: It allows users to extract valuable information and content from PDFs efficiently. This can be particularly useful for researchers and professionals who need to extract data, figures, or text from scientific papers.

3. **Metadata Extraction**: Nougat can also extract metadata from PDF documents, providing essential details about the document, such as title, author, and publication date.

## Parameters

- `model_name_or_path` (str): The name or path of the pretrained Nougat model. Default: "facebook/nougat-base".
- `min_length` (int): The minimum length of the generated transcription. Default: 1.
- `max_new_tokens` (int): The maximum number of new tokens to generate in the Markdown transcription. Default: 30.

## Usage

To use Nougat, follow these steps:

1. Initialize the Nougat instance:

```python
from swarms.models import Nougat

nougat = Nougat()
```

### Example 1 - Initialization

```python
nougat = Nougat()
```

2. Transcribe a PDF image using Nougat:

```python
markdown_transcription = nougat("path/to/pdf_file.png")
```

### Example 2 - PDF Transcription

```python
nougat = Nougat()
markdown_transcription = nougat("path/to/pdf_file.png")
```

3. Extract information from a PDF:

```python
information = nougat.extract_information("path/to/pdf_file.png")
```

### Example 3 - Information Extraction

```python
nougat = Nougat()
information = nougat.extract_information("path/to/pdf_file.png")
```

4. Extract metadata from a PDF:

```python
metadata = nougat.extract_metadata("path/to/pdf_file.png")
```

### Example 4 - Metadata Extraction

```python
nougat = Nougat()
metadata = nougat.extract_metadata("path/to/pdf_file.png")
```

## How Nougat Works

Nougat employs a vision encoder-decoder model, along with a dedicated processor, to transcribe PDFs into Markdown format and perform information and metadata extraction. Here's how it works:

1. **Initialization**: When you create a Nougat instance, you can specify the model to use, the minimum transcription length, and the maximum number of new tokens to generate.

2. **Processing PDFs**: Nougat can process PDFs as input. You can provide the path to a PDF document.

3. **Image Processing**: The processor converts PDF pages into images, which are then encoded by the model.

4. **Transcription**: Nougat generates Markdown transcriptions of PDF content, ensuring a minimum length and respecting the token limit.

5. **Information Extraction**: Information extraction involves parsing the Markdown transcription to identify key details or content of interest.

6. **Metadata Extraction**: Metadata extraction involves identifying and extracting document metadata, such as title, author, and publication date.

## Additional Information

- Nougat leverages the "facebook/nougat-base" pretrained model, which is specifically designed for document transcription and extraction tasks.
- You can adjust the minimum transcription length and the maximum number of new tokens to control the output's length and quality.
- Nougat can be run on both CPU and GPU devices.

That concludes the documentation for Nougat. We hope you find this tool valuable for your PDF transcription, information extraction, and metadata extraction needs. If you have any questions or encounter any issues, please refer to the Nougat documentation for further assistance. Enjoy using Nougat!