## BingChat User Guide

Welcome to the BingChat user guide! This document provides a step-by-step tutorial on how to leverage the BingChat class, an interface to the EdgeGPT model by OpenAI.

### Table of Contents
1. [Installation & Prerequisites](#installation)
2. [Setting Up BingChat](#setup)
3. [Interacting with BingChat](#interacting)
4. [Generating Images](#images)
5. [Managing Cookies](#cookies)

### Installation & Prerequisites <a name="installation"></a>

Before initializing the BingChat model, ensure you have the necessary dependencies installed:

```shell
pip install EdgeGPT
```

Additionally, you must have a `cookies.json` file which is necessary for authenticating with EdgeGPT.

### Setting Up BingChat <a name="setup"></a>

To start, import the BingChat class:

```python
from bing_chat import BingChat
```

Initialize BingChat with the path to your `cookies.json`:

```python
chat = BingChat(cookies_path="./path/to/cookies.json")
```

### Interacting with BingChat <a name="interacting"></a>

You can obtain text responses from the EdgeGPT model by simply calling the instantiated object:

```python
response = chat("Hello, my name is ChatGPT")
print(response)
```

You can also specify the conversation style:

```python
from bing_chat import ConversationStyle

response = chat("Tell me a joke", style=ConversationStyle.creative)
print(response)
```

### Generating Images <a name="images"></a>

BingChat allows you to generate images based on text prompts:

```python
image_path = chat.create_img("Sunset over mountains", auth_cookie="YOUR_AUTH_COOKIE")
print(f"Image saved at: {image_path}")
```

Ensure you provide the required `auth_cookie` for image generation.

### Managing Cookies <a name="cookies"></a>

You can set a directory path for managing cookies using the `set_cookie_dir_path` method:

BingChat.set_cookie_dir_path("./path/to/cookies_directory")


