# llama3Hosted Documentation

## Overview

The `llama3Hosted` class is a high-level interface for interacting with a hosted version of the Llama3 model. This class is designed to simplify the process of generating responses from the Llama3 model by providing an easy-to-use interface for sending requests and receiving responses. The Llama3 model is a state-of-the-art language model developed by Meta, known for its ability to generate human-like text based on the input it receives.

### Key Features

- **Model Customization**: Allows the user to specify which version of the Llama3 model to use.
- **Temperature Control**: Adjusts the randomness of the generated responses.
- **Token Limitation**: Sets a limit on the maximum number of tokens in the generated response.
- **System Prompt**: Defines the initial context for the conversation, guiding the model's responses.

## Purpose

The `llama3Hosted` class is designed to provide developers with a straightforward way to utilize the capabilities of the Llama3 model without dealing with the complexities of model hosting and API management. It is particularly useful for applications that require natural language understanding and generation, such as chatbots, virtual assistants, and content generation tools.

## Class Definition

### llama3Hosted Parameters

| Parameter      | Type   | Default                                 | Description                                                  |
|----------------|--------|-----------------------------------------|--------------------------------------------------------------|
| `model`        | `str`  | `"meta-llama/Meta-Llama-3-8B-Instruct"` | The name or path of the Llama3 model to use.                 |
| `temperature`  | `float`| `0.8`                                   | The temperature parameter for generating responses.          |
| `max_tokens`   | `int`  | `4000`                                  | The maximum number of tokens in the generated response.      |
| `system_prompt`| `str`  | `"You are a helpful assistant."`        | The system prompt to use for generating responses.           |
| `*args`        |        |                                         | Variable length argument list.                               |
| `**kwargs`     |        |                                         | Arbitrary keyword arguments.                                 |

### Attributes

| Attribute      | Type   | Description                                                  |
|----------------|--------|--------------------------------------------------------------|
| `model`        | `str`  | The name or path of the Llama3 model.                        |
| `temperature`  | `float`| The temperature parameter for generating responses.          |
| `max_tokens`   | `int`  | The maximum number of tokens in the generated response.      |
| `system_prompt`| `str`  | The system prompt for generating responses.                  |

## Method: run

### Parameters

| Parameter | Type   | Description                       |
|-----------|--------|-----------------------------------|
| `task`    | `str`  | The user's task or input.         |
| `*args`   |        | Variable length argument list.    |
| `**kwargs`|        | Arbitrary keyword arguments.      |

### Returns

| Type | Description                                |
|------|--------------------------------------------|
| `str`| The generated response from the Llama3 model.|

### Usage Examples
First install weather_swarm with:

`$ pip install -U weather-swarm`


#### Example 1: Basic Usage

```python
from weather_swarmn import llama3Hosted

llama = llama3Hosted()
response = llama.run("Tell me a joke.")
print(response)
```

#### Example 2: Custom Model and Parameters

```python
import requests
import json
from weather_swarmn import llama3Hosted


llama = llama3Hosted(
    model="custom-llama-model",
    temperature=0.5,
    max_tokens=2000,
    system_prompt="You are a witty assistant."
)
response = llama.run("What's the weather like today?")
print(response)
```

#### Example 3: Using Additional Arguments

```python
from weather_swarmn import llama3Hosted

llama = llama3Hosted()
response = llama.run("Write a short story.", custom_stop_tokens=[128002, 128003])
print(response)
```

## Additional Information and Tips

- **Temperature Parameter**: The temperature parameter controls the randomness of the model's output. Lower values (close to 0) make the output more deterministic, while higher values (up to 1) make it more random.
- **System Prompt**: Crafting an effective system prompt can significantly impact the quality and relevance of the model's responses. Ensure the prompt aligns well with the intended use case.
- **Error Handling**: Always include error handling when making API requests to ensure your application can gracefully handle any issues that arise.

## References and Resources

- [Llama3 Model Documentation](https://github.com/facebookresearch/llama)
- [Requests Library Documentation](https://docs.python-requests.org/en/latest/)
- [JSON Library Documentation](https://docs.python.org/3/library/json.html)

This documentation provides a comprehensive overview of the `llama3Hosted` class, its parameters, attributes, methods, and usage examples. By following this guide, developers can effectively integrate and utilize the Llama3 model in their applications.