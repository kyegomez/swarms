# Fuyu Documentation

## Introduction

Welcome to the documentation for Fuyu, a versatile model for generating text conditioned on both textual prompts and images. Fuyu is based on the Adept's Fuyu model and offers a convenient way to create text that is influenced by the content of an image. In this documentation, you will find comprehensive information on the Fuyu class, its architecture, usage, and examples.

## Overview

Fuyu is a text generation model that leverages both text and images to generate coherent and contextually relevant text. It combines state-of-the-art language modeling techniques with image processing capabilities to produce text that is semantically connected to the content of an image. Whether you need to create captions for images or generate text that describes visual content, Fuyu can assist you.

## Class Definition

```python
class Fuyu:
    def __init__(
        self,
        pretrained_path: str = "adept/fuyu-8b",
        device_map: str = "cuda:0",
        max_new_tokens: int = 7,
    ):
```

## Purpose

The Fuyu class serves as a convenient interface for using the Adept's Fuyu model. It allows you to generate text based on a textual prompt and an image. The primary purpose of Fuyu is to provide a user-friendly way to create text that is influenced by visual content, making it suitable for various applications, including image captioning, storytelling, and creative text generation.

## Parameters

- `pretrained_path` (str): The path to the pretrained Fuyu model. By default, it uses the "adept/fuyu-8b" model.
- `device_map` (str): The device to use for model inference (e.g., "cuda:0" for GPU or "cpu" for CPU). Default: "cuda:0".
- `max_new_tokens` (int): The maximum number of tokens to generate in the output text. Default: 7.

## Usage

To use Fuyu, follow these steps:

1. Initialize the Fuyu instance:

```python
from swarms.models.fuyu import Fuyu

fuyu = Fuyu()
```


2. Generate Text with Fuyu:

```python
text = "Hello, my name is"
img_path = "path/to/image.png"
output_text = fuyu(text, img_path)
```

### Example 2 - Text Generation

```python
from swarms.models.fuyu import Fuyu

fuyu = Fuyu()

text = "Hello, my name is"

img_path = "path/to/image.png"

output_text = fuyu(text, img_path)
print(output_text)
```

## How Fuyu Works

Fuyu combines text and image processing to generate meaningful text outputs. Here's how it works:

1. **Initialization**: When you create a Fuyu instance, you specify the pretrained model path, the device for inference, and the maximum number of tokens to generate.

2. **Processing Text and Images**: Fuyu can process both textual prompts and images. You provide a text prompt and the path to an image as input.

3. **Tokenization**: Fuyu tokenizes the input text and encodes the image using its tokenizer.

4. **Model Inference**: The model takes the tokenized inputs and generates text that is conditioned on both the text and the image.

5. **Output Text**: Fuyu returns the generated text as the output.

## Additional Information

- Fuyu uses the Adept's Fuyu model, which is pretrained on a large corpus of text and images, making it capable of generating coherent and contextually relevant text.
- You can specify the device for inference to utilize GPU acceleration if available.
- The `max_new_tokens` parameter allows you to control the length of the generated text.

That concludes the documentation for Fuyu. We hope you find this model useful for your text generation tasks that involve images. If you have any questions or encounter any issues, please refer to the Fuyu documentation for further assistance. Enjoy working with Fuyu!