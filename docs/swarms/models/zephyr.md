# `Zephyr` Documentation

## Introduction

Welcome to the documentation for Zephyr, a language model by Hugging Face designed for text generation tasks. Zephyr is capable of generating text in response to prompts and is highly customizable using various parameters. This document will provide you with a detailed understanding of Zephyr, its purpose, and how to effectively use it in your projects.

## Overview

Zephyr is a text generation model that can be used to generate human-like text based on a given prompt. It utilizes the power of transformers and fine-tuning to create coherent and contextually relevant text. Users can control the generated text's characteristics through parameters such as `temperature`, `top_k`, `top_p`, and `max_new_tokens`.

## Class Definition

```python
class Zephyr:
    def __init__(
        self,
        max_new_tokens: int = 300,
        temperature: float = 0.5,
        top_k: float = 50,
        top_p: float = 0.95,
    ):
        """
        Initialize the Zephyr model.

        Args:
            max_new_tokens (int): The maximum number of tokens in the generated text.
            temperature (float): The temperature parameter, controlling the randomness of the output.
            top_k (float): The top-k parameter, limiting the vocabulary used in generation.
            top_p (float): The top-p parameter, controlling the diversity of the output.
        """
```

## Parameters

- `max_new_tokens` (int): The maximum number of tokens in the generated text.
- `temperature` (float): The temperature parameter, controlling the randomness of the output.
- `top_k` (float): The top-k parameter, limiting the vocabulary used in generation.
- `top_p` (float): The top-p parameter, controlling the diversity of the output.

## Usage

To use the Zephyr model, follow these steps:

1. Initialize the Zephyr model with your desired parameters:

```python
from swarms.models import Zephyr

model = Zephyr(max_new_tokens=300, temperature=0.7, top_k=50, top_p=0.95)
```

2. Generate text by providing a prompt:

```python
output = model("Generate a funny joke about cats")
print(output)
```

### Example 1 - Generating a Joke

```python
model = Zephyr(max_new_tokens=100)
output = model("Tell me a joke about programmers")
print(output)
```

### Example 2 - Writing Poetry

```python
model = Zephyr(temperature=0.2, top_k=30)
output = model("Write a short poem about the moon")
print(output)
```

### Example 3 - Asking for Advice

```python
model = Zephyr(temperature=0.8, top_p=0.9)
output = model("Give me advice on starting a healthy lifestyle")
print(output)
```

## Additional Information

- Zephyr is based on the Hugging Face Transformers library and uses the "HuggingFaceH4/zephyr-7b-alpha" model.
- The generated text can vary based on the values of `temperature`, `top_k`, and `top_p`. Experiment with these parameters to achieve the desired output.
- The `max_new_tokens` parameter can be adjusted to control the length of the generated text.
- You can integrate Zephyr into chat applications, creative writing projects, or any task that involves generating human-like text.

That concludes the documentation for Zephyr. We hope you find this model useful for your text generation needs! If you have any questions or encounter any issues, please refer to the Hugging Face Transformers documentation for further assistance. Happy text generation!