# pdf_to_text

## Introduction
The function `pdf_to_text` is a Python utility for converting a PDF file into a string of text content. It leverages the `pypdf` library, an excellent Python library for processing PDF files. The function takes in a PDF file's path and reads its content, subsequently returning the extracted textual data.

This function can be very useful when you want to extract textual information from PDF files automatically. For instance, when processing a large number of documents, performing textual analysis, or when you're dealing with text data that is only available in PDF format.

## Class / Function Definition

`pdf_to_text` is a standalone function defined as follows:

```python
def pdf_to_text(pdf_path: str) -> str:
```

## Parameters

| Parameter   | Type  |  Description  | 
|:-:|---|---|
|    pdf_path   |   str |   The path to the PDF file to be converted  |

## Returns

| Return Value   | Type  |  Description  | 
|:-:|---|---|
|    text   |   str |   The text extracted from the PDF file.  |

## Raises

| Exception  |  Description  | 
|---|---|
|   FileNotFoundError  |  If the PDF file is not found at the specified path. |
|   Exception  |  If there is an error in reading the PDF file. |

## Function Description 

`pdf_to_text` utilises the `PdfReader` function from the `pypdf` library to read the PDF file. If the PDF file does not exist at the specified path or there was an error while reading the file, appropriate exceptions will be raised. It then iterates through each page in the PDF and uses the `extract_text` function to extract the text content from each page. These contents are then concatenated into a single variable and returned as the result.

## Usage Examples

To use this function, you first need to install the `pypdf` library. It can be installed via pip:

```python
!pip install pypdf
```

Then, you should import the `pdf_to_text` function:

```python
from swarms.utils import pdf_to_text
```

Here is an example of how to use `pdf_to_text`:

```python
# Define the path to the pdf file
pdf_path = 'sample.pdf'

# Use the function to extract text
text = pdf_to_text(pdf_path)

# Print the extracted text
print(text)
```

## Tips and Additional Information
- Ensure that the PDF file path is valid and that the file exists at the specified location. If the file does not exist, a `FileNotFoundError` will be raised.
- This function reads the text from the PDF. It does not handle images, graphical elements, or any non-text content.
- If the PDF contains scanned images rather than textual data, the `extract_text` function may not be able to extract any text. In such cases, you would require OCR (Optical Character Recognition) tools to extract the text. 
- Be aware of the possibility that the output string might contain special characters or escape sequences because they were part of the PDF's content. You might need to clean the resulting text according to your requirements.
- The function uses the pypdf library to facilitate the PDF reading and text extraction. For any issues related to PDF manipulation, consult the [pypdf library documentation](https://pythonhosted.org/pypdf/).
