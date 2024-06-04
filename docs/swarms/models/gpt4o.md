# Documentation for GPT4o Module

## Overview and Introduction

The `GPT4o` module is a multi-modal conversational model based on OpenAI's GPT-4 architecture. It extends the functionality of the `BaseMultiModalModel` class, enabling it to handle both text and image inputs for generating diverse and contextually rich responses. This module leverages the power of the GPT-4 model to enhance interactions by integrating visual information with textual prompts, making it highly relevant for applications requiring multi-modal understanding and response generation.

### Key Concepts
- **Multi-Modal Model**: A model that can process and generate responses based on multiple types of inputs, such as text and images.
- **System Prompt**: A predefined prompt to guide the conversation flow.
- **Temperature**: A parameter that controls the randomness of the response generation.
- **Max Tokens**: The maximum number of tokens (words or word pieces) in the generated response.

## Class Definition

### `GPT4o` Class


### Parameters

| Parameter       | Type   | Description                                                                          |
|-----------------|--------|--------------------------------------------------------------------------------------|
| `system_prompt` | `str`  | The system prompt to be used in the conversation.                                     |
| `temperature`   | `float`| The temperature parameter for generating diverse responses. Default is `0.1`.        |
| `max_tokens`    | `int`  | The maximum number of tokens in the generated response. Default is `300`.            |
| `openai_api_key`| `str`  | The API key for accessing the OpenAI GPT-4 API.                                       |
| `*args`         |        | Additional positional arguments.                                                     |
| `**kwargs`      |        | Additional keyword arguments.                                                        |

## Functionality and Usage

### `encode_image` Function

The `encode_image` function is used to encode an image file into a base64 string format, which can then be included in the request to the GPT-4 API.

#### Parameters

| Parameter     | Type   | Description                                  |
|---------------|--------|----------------------------------------------|
| `image_path`  | `str`  | The local path to the image file to be encoded. |

#### Returns

| Return Type | Description                     |
|-------------|---------------------------------|
| `str`       | The base64 encoded string of the image. |

### `GPT4o.__init__` Method

The constructor for the `GPT4o` class initializes the model with the specified parameters and sets up the OpenAI client.

### `GPT4o.run` Method

The `run` method executes the GPT-4o model to generate a response based on the provided task and optional image.

#### Parameters

| Parameter     | Type   | Description                                        |
|---------------|--------|----------------------------------------------------|
| `task`        | `str`  | The task or user prompt for the conversation.      |
| `local_img`   | `str`  | The local path to the image file.                  |
| `img`         | `str`  | The URL of the image.                              |
| `*args`       |        | Additional positional arguments.                   |
| `**kwargs`    |        | Additional keyword arguments.                      |

#### Returns

| Return Type | Description                                      |
|-------------|--------------------------------------------------|
| `str`       | The generated response from the GPT-4o model.    |

## Usage Examples

### Example 1: Basic Text Prompt

```python
from swarms import GPT4o

# Initialize the model
model = GPT4o(
    system_prompt="You are a helpful assistant.",
    temperature=0.7,
    max_tokens=150,
    openai_api_key="your_openai_api_key"
)

# Define the task
task = "What is the capital of France?"

# Generate response
response = model.run(task)
print(response)
```

### Example 2: Text Prompt with Local Image

```python
from swarms import GPT4o

# Initialize the model
model = GPT4o(
    system_prompt="Describe the image content.",
    temperature=0.5,
    max_tokens=200,
    openai_api_key="your_openai_api_key"
)

# Define the task and image path
task = "Describe the content of this image."
local_img = "path/to/your/image.jpg"

# Generate response
response = model.run(task, local_img=local_img)
print(response)
```

### Example 3: Text Prompt with Image URL

```python
from swarms import GPT4o

# Initialize the model
model = GPT4o(
    system_prompt="You are a visual assistant.",
    temperature=0.6,
    max_tokens=250,
    openai_api_key="your_openai_api_key"
)

# Define the task and image URL
task = "What can you tell about the scenery in this image?"
img_url = "http://example.com/image.jpg"

# Generate response
response = model.run(task, img=img_url)
print(response)
```

## Additional Information and Tips

- **API Key Management**: Ensure that your OpenAI API key is securely stored and managed. Do not hard-code it in your scripts. Use environment variables or secure storage solutions.
- **Image Encoding**: The `encode_image` function is crucial for converting images to a base64 format suitable for API requests. Ensure that the images are accessible and properly formatted.
- **Temperature Parameter**: Adjust the `temperature` parameter to control the creativity of the model's responses. Lower values make the output more deterministic, while higher values increase randomness.
- **Token Limit**: Be mindful of the `max_tokens` parameter to avoid exceeding the API's token limits. This parameter controls the length of the generated responses.

## References and Resources

- [OpenAI API Documentation](https://beta.openai.com/docs/)
- [Python Base64 Encoding](https://docs.python.org/3/library/base64.html)
- [dotenv Documentation](https://saurabh-kumar.com/python-dotenv/)
- [BaseMultiModalModel Documentation](https://swarms.apac.ai)