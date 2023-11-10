# `GPT4Vision` Documentation

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Initialization](#initialization)
- [Methods](#methods)
  - [process_img](#process_img)
  - [__call__](#__call__)
  - [run](#run)
  - [arun](#arun)
- [Configuration Options](#configuration-options)
- [Usage Examples](#usage-examples)
- [Additional Tips](#additional-tips)
- [References and Resources](#references-and-resources)

---

## Overview

The GPT4Vision Model API is designed to provide an easy-to-use interface for interacting with the OpenAI GPT-4 Vision model. This model can generate textual descriptions for images and answer questions related to visual content. Whether you want to describe images or perform other vision-related tasks, GPT4Vision makes it simple and efficient.

The library offers a straightforward way to send images and tasks to the GPT-4 Vision model and retrieve the generated responses. It handles API communication, authentication, and retries, making it a powerful tool for developers working with computer vision and natural language processing tasks.

## Installation

To use the GPT4Vision Model API, you need to install the required dependencies and configure your environment. Follow these steps to get started:

1. Install the required Python package:

   ```bash
   pip3 install --upgrade swarms
   ```

2. Make sure you have an OpenAI API key. You can obtain one by signing up on the [OpenAI platform](https://beta.openai.com/signup/).

3. Set your OpenAI API key as an environment variable. You can do this in your code or your environment configuration. Alternatively, you can provide the API key directly when initializing the `GPT4Vision` class.

## Initialization

To start using the GPT4Vision Model API, you need to create an instance of the `GPT4Vision` class. You can customize its behavior by providing various configuration options, but it also comes with sensible defaults.

Here's how you can initialize the `GPT4Vision` class:

```python
from swarms.models.gpt4v import GPT4Vision

gpt4vision = GPT4Vision(
    api_key="Your Key"
)
```

The above code initializes the `GPT4Vision` class with default settings. You can adjust these settings as needed.

## Methods

### `process_img`

The `process_img` method is used to preprocess an image before sending it to the GPT-4 Vision model. It takes the image path as input and returns the processed image in a format suitable for API requests.

```python
processed_img = gpt4vision.process_img(img_path)
```

- `img_path` (str): The file path or URL of the image to be processed.

### `__call__`

The `__call__` method is the main method for interacting with the GPT-4 Vision model. It sends the image and tasks to the model and returns the generated response.

```python
response = gpt4vision(img, tasks)
```

- `img` (Union[str, List[str]]): Either a single image URL or a list of image URLs to be used for the API request.
- `tasks` (List[str]): A list of tasks or questions related to the image(s).

This method returns a `GPT4VisionResponse` object, which contains the generated answer.

### `run`

The `run` method is an alternative way to interact with the GPT-4 Vision model. It takes a single task and image URL as input and returns the generated response.

```python
response = gpt4vision.run(task, img)
```

- `task` (str): The task or question related to the image.
- `img` (str): The image URL to be used for the API request.

This method simplifies interactions when dealing with a single task and image.

### `arun`

The `arun` method is an asynchronous version of the `run` method. It allows for asynchronous processing of API requests, which can be useful in certain scenarios.

```python
import asyncio

async def main():
    response = await gpt4vision.arun(task, img)
    print(response)

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
```

- `task` (str): The task or question related to the image.
- `img` (str): The image URL to be used for the API request.

## Configuration Options

The `GPT4Vision` class provides several configuration options that allow you to customize its behavior:

- `max_retries` (int): The maximum number of retries to make to the API. Default: 3
- `backoff_factor` (float): The backoff factor to use for exponential backoff. Default: 2.0
- `timeout_seconds` (int): The timeout in seconds for the API request. Default: 10
- `api_key` (str): The API key to use for the API request. Default: None (set via environment variable)
- `quality` (str): The quality of the image to generate. Options: 'low' or 'high'. Default: 'low'
- `max_tokens` (int): The maximum number of tokens to use for the API request. Default: 200

## Usage Examples

### Example 1: Generating Image Descriptions

```python
gpt4vision = GPT4Vision()
img = "https://example.com/image.jpg"
tasks = ["Describe this image."]
response = gpt4vision(img, tasks)
print(response.answer)
```

In this example, we create an instance of `GPT4Vision`, provide an image URL, and ask the model to describe the image. The response contains the generated description.

### Example 2: Custom Configuration

```python
custom_config = {
    "max_retries": 5,
    "timeout_seconds": 20,
    "quality": "high",
    "max_tokens": 300,
}
gpt4vision = GPT4Vision(**custom_config)
img = "https://example.com/another_image.jpg"
tasks = ["What objects can you identify in this image?"]
response = gpt4vision(img, tasks)
print(response.answer)
```

In this example, we create an instance of `GPT4Vision` with custom configuration options. We set a higher timeout, request high-quality images, and allow more tokens in the response.

### Example 3: Using the `run` Method

```python
gpt4vision = GPT4Vision()
img = "https://example.com/image.jpg"
task = "Describe this image in detail."
response = gpt4vision.run(task, img)
print(response)
```

In this example, we use the `run` method to simplify the interaction by providing a single task and image URL.

# Model Usage and Image Understanding

The GPT-4 Vision model processes images in a unique way, allowing it to answer questions about both or each of the images independently. Here's an overview:

| Purpose                                 | Description                                                                                                      |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Image Understanding                      | The model is shown two copies of the same image and can answer questions about both or each of the images independently. |

# Image Detail Control

You have control over how the model processes the image and generates textual understanding by using the `detail` parameter, which has two options: `low` and `high`.

| Detail   | Description                                                                                                                                                                              |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| low      | Disables the "high-res" model. The model receives a low-res 512 x 512 version of the image and represents the image with a budget of 65 tokens. Ideal for use cases not requiring high detail. |
| high     | Enables "high-res" mode. The model first sees the low-res image and then creates detailed crops of input images as 512px squares based on the input image size. Uses a total of 129 tokens.  |

# Managing Images

To use the Chat Completions API effectively, you must manage the images you pass to the model. Here are some key considerations:

| Management Aspect        | Description                                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------------------- |
| Image Reuse              | To pass the same image multiple times, include the image with each API request.                  |
| Image Size Optimization   | Improve latency by downsizing images to meet the expected size requirements.                    |
| Image Deletion           | After processing, images are deleted from OpenAI servers and not retained. No data is used for training. |

# Limitations

While GPT-4 with Vision is powerful, it has some limitations:

| Limitation                                   | Description                                                                                         |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Medical Images                               | Not suitable for interpreting specialized medical images like CT scans.                                |
| Non-English Text                            | May not perform optimally when handling non-Latin alphabets, such as Japanese or Korean.               |
| Large Text in Images                        | Enlarge text within images for readability, but avoid cropping important details.                       |
| Rotated or Upside-Down Text/Images          | May misinterpret rotated or upside-down text or images.                                                  |
| Complex Visual Elements                     | May struggle to understand complex graphs or text with varying colors or styles.                        |
| Spatial Reasoning                            | Struggles with tasks requiring precise spatial localization, such as identifying chess positions.       |
| Accuracy                                     | May generate incorrect descriptions or captions in certain scenarios.                                    |
| Panoramic and Fisheye Images                | Struggles with panoramic and fisheye images.                                                              |

# Calculating Costs

Image inputs are metered and charged in tokens. The token cost depends on the image size and detail option.

| Example                                       | Token Cost  |
| --------------------------------------------- | ----------- |
| 1024 x 1024 square image in detail: high mode | 765 tokens  |
| 2048 x 4096 image in detail: high mode        | 1105 tokens |
| 4096 x 8192 image in detail: low mode         | 85 tokens   |

# FAQ

Here are some frequently asked questions about GPT-4 with Vision:

| Question                                     | Answer                                                                                             |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Fine-Tuning Image Capabilities                | No, fine-tuning the image capabilities of GPT-4 is not supported at this time.                         |
| Generating Images                            | GPT-4 is used for understanding images, not generating them.                                            |
| Supported Image File Types                   | Supported image file types include PNG (.png), JPEG (.jpeg and .jpg), WEBP (.webp), and non-animated GIF (.gif). |
| Image Size Limitations                       | Image uploads are restricted to 20MB per image.                                                           |
| Image Deletion                               | Uploaded images are automatically deleted after processing by the model.                                   |
| Learning More                               | For more details about GPT-4 with Vision, refer to the GPT-4 with Vision system card.                      |
| CAPTCHA Submission                           | CAPTCHAs are blocked for safety reasons.                                                                  |
| Rate Limits                                  | Image processing counts toward your tokens per minute (TPM) limit. Refer to the calculating costs section for details. |
| Image Metadata                               | The model does not receive image metadata.                                                                |
| Handling Unclear Images                      | If an image is unclear, the model will do its best to interpret it, but results may be less accurate.   |



## Additional Tips

- Make sure to handle potential exceptions and errors when making API requests. The library includes retries and error handling, but it's essential to handle exceptions gracefully in your code.
- Experiment with different configuration options to optimize the trade-off between response quality and response time based on your specific requirements.

## References and Resources

- [OpenAI Platform](https://beta.openai.com/signup/): Sign up for an OpenAI API key.
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference/chat/create): Official API documentation for the GPT-4 Vision model.

Now you have a comprehensive understanding of the GPT4Vision Model API, its configuration options, and how to use it for various computer vision and natural language processing tasks. Start experimenting and integrating it into your projects to leverage the power of GPT-4 Vision for image-related tasks.

# Conclusion

With GPT-4 Vision, you have a powerful tool for understanding and generating textual descriptions for images. By considering its capabilities, limitations, and cost calculations, you can effectively leverage this model for various image-related tasks.