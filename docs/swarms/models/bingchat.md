# BingChat Documentation

## Introduction

Welcome to the documentation for BingChat, a powerful chatbot and image generation tool based on OpenAI's GPT model. This documentation provides a comprehensive understanding of BingChat, its architecture, usage, and how it can be integrated into your projects.

## Overview

BingChat is designed to provide text responses and generate images based on user prompts. It utilizes the capabilities of the GPT model to generate creative and contextually relevant responses. In addition, it can create images based on text prompts, making it a versatile tool for various applications.

## Class Definition

```python
class BingChat:
    def __init__(self, cookies_path: str):
```

## Usage

To use BingChat, follow these steps:

1. Initialize the BingChat instance:

```python
from swarms.models.bing_chat import BingChat

edgegpt = BingChat(cookies_path="./path/to/cookies.json")
```

2. Get a text response:

```python
response = edgegpt("Hello, my name is ChatGPT")
print(response)
```

### Example 1 - Text Response

```python
from swarms.models.bing_chat import BingChat

edgegpt = BingChat(cookies_path="./path/to/cookies.json")
response = edgegpt("Hello, my name is ChatGPT")
print(response)
```

3. Generate an image based on a text prompt:

```python
image_path = edgegpt.create_img(
    "Sunset over mountains", output_dir="./output", auth_cookie="your_auth_cookie"
)
print(f"Generated image saved at {image_path}")
```

### Example 2 - Image Generation

```python
from swarms.models.bing_chat import BingChat

edgegpt = BingChat(cookies_path="./path/to/cookies.json")

image_path = edgegpt.create_img(
    "Sunset over mountains", output_dir="./output", auth_cookie="your_auth_cookie"
)

print(f"Generated image saved at {image_path}")
```

4. Set the directory path for managing cookies:

```python
BingChat.set_cookie_dir_path("./cookie_directory")
```

### Example 3 - Set Cookie Directory Path

```python
BingChat.set_cookie_dir_path("./cookie_directory")
```

## How BingChat Works

BingChat works by utilizing cookies for authentication and interacting with OpenAI's GPT model. Here's how it works:

1. **Initialization**: When you create a BingChat instance, it loads the necessary cookies for authentication with BingChat.

2. **Text Response**: You can use the `__call__` method to get a text response from the GPT model based on the provided prompt. You can specify the conversation style for different response types.

3. **Image Generation**: The `create_img` method allows you to generate images based on text prompts. It requires an authentication cookie and saves the generated images to the specified output directory.

4. **Cookie Directory**: You can set the directory path for managing cookies using the `set_cookie_dir_path` method.

## Parameters

- `cookies_path`: The path to the cookies.json file necessary for authenticating with BingChat.

## Additional Information

- BingChat provides both text-based and image-based responses, making it versatile for various use cases.
- Cookies are used for authentication, so make sure to provide the correct path to the cookies.json file.
- Image generation requires an authentication cookie, and the generated images can be saved to a specified directory.

That concludes the documentation for BingChat. We hope you find this tool valuable for your text generation and image generation tasks. If you have any questions or encounter any issues, please refer to the BingChat documentation for further assistance. Enjoy working with BingChat!