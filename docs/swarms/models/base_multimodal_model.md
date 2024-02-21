# `BaseMultiModalModel` Documentation

Swarms is a Python library that provides a framework for running multimodal AI models. It allows you to combine text and image inputs and generate coherent and context-aware responses. This library is designed to be extensible, allowing you to integrate various multimodal models.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [BaseMultiModalModel Class](#basemultimodalmodel-class)
    - [Initialization](#initialization)
    - [Methods](#methods)
5. [Usage Examples](#usage-examples)
6. [Additional Tips](#additional-tips)
7. [References and Resources](#references-and-resources)

## 1. Introduction <a name="introduction"></a>

Swarms is designed to simplify the process of working with multimodal AI models. These models are capable of understanding and generating content based on both textual and image inputs. With this library, you can run such models and receive context-aware responses.

## 2. Installation <a name="installation"></a>

To install swarms, you can use pip:

```bash
pip install swarms
```

## 3. Getting Started <a name="getting-started"></a>

To get started with Swarms, you'll need to import the library and create an instance of the `BaseMultiModalModel` class. This class serves as the foundation for running multimodal models.

```python
from swarms.models import BaseMultiModalModel

model = BaseMultiModalModel(
    model_name="your_model_name",
    temperature=0.5,
    max_tokens=500,
    max_workers=10,
    top_p=1,
    top_k=50,
    beautify=False,
    device="cuda",
    max_new_tokens=500,
    retries=3,
)
```

You can customize the initialization parameters based on your model's requirements.

## 4. BaseMultiModalModel Class <a name="basemultimodalmodel-class"></a>

### Initialization <a name="initialization"></a>

The `BaseMultiModalModel` class is initialized with several parameters that control its behavior. Here's a breakdown of the initialization parameters:

| Parameter        | Description                                                                                           | Default Value |
|------------------|-------------------------------------------------------------------------------------------------------|---------------|
| `model_name`     | The name of the multimodal model to use.                                                              | None          |
| `temperature`    | The temperature parameter for controlling randomness in text generation.                            | 0.5           |
| `max_tokens`     | The maximum number of tokens in the generated text.                                                    | 500           |
| `max_workers`    | The maximum number of concurrent workers for running tasks.                                           | 10            |
| `top_p`          | The top-p parameter for filtering words in text generation.                                            | 1             |
| `top_k`          | The top-k parameter for filtering words in text generation.                                            | 50            |
| `beautify`       | Whether to beautify the output text.                                                                  | False         |
| `device`         | The device to run the model on (e.g., 'cuda' or 'cpu').                                                | 'cuda'        |
| `max_new_tokens` | The maximum number of new tokens allowed in generated responses.                                       | 500           |
| `retries`        | The number of retries in case of an error during text generation.                                      | 3             |
| `system_prompt`  | A system-level prompt to set context for generation.                                                   | None          |
| `meta_prompt`    | A meta prompt to provide guidance for including image labels in responses.                             | None          |

### Methods <a name="methods"></a>

The `BaseMultiModalModel` class defines various methods for running multimodal models and managing interactions:

- `run(task: str, img: str) -> str`: Run the multimodal model with a text task and an image URL to generate a response.

- `arun(task: str, img: str) -> str`: Run the multimodal model asynchronously with a text task and an image URL to generate a response.

- `get_img_from_web(img: str) -> Image`: Fetch an image from a URL and return it as a PIL Image.

- `encode_img(img: str) -> str`: Encode an image to base64 format.

- `get_img(img: str) -> Image`: Load an image from the local file system and return it as a PIL Image.

- `clear_chat_history()`: Clear the chat history maintained by the model.

- `run_many(tasks: List[str], imgs: List[str]) -> List[str]`: Run the model on multiple text tasks and image URLs concurrently and return a list of responses.

- `run_batch(tasks_images: List[Tuple[str, str]]) -> List[str]`: Process a batch of text tasks and image URLs and return a list of responses.

- `run_batch_async(tasks_images: List[Tuple[str, str]]) -> List[str]`: Process a batch of text tasks and image URLs asynchronously and return a list of responses.

- `run_batch_async_with_retries(tasks_images: List[Tuple[str, str]]) -> List[str]`: Process a batch of text tasks and image URLs asynchronously with retries in case of errors and return a list of responses.

- `unique_chat_history() -> List[str]`: Get the unique chat history stored by the model.

- `run_with_retries(task: str, img: str) -> str`: Run the model with retries in case of an error.

- `run_batch_with_retries(tasks_images: List[Tuple[str, str]]) -> List[str]`: Run a batch of tasks with retries in case of errors and return a list of responses.

- `_tokens_per_second() -> float`: Calculate the tokens generated per second during text generation.

- `_time_for_generation(task: str) -> float`: Measure the time taken for text generation for a specific task.

- `generate_summary(text: str) -> str`: Generate a summary of the provided text.

- `set_temperature(value: float)`: Set the temperature parameter for controlling randomness in text generation.

- `set_max_tokens(value: int)`: Set the maximum number of tokens allowed in generated responses.

- `get_generation_time() -> float`: Get the time taken for text generation for the last task.

- `get_chat_history() -> List[str]`: Get the chat history, including all interactions.

- `get_unique_chat_history() -> List[str]`: Get the unique chat history, removing duplicate interactions.

- `get_chat_history_length() -> int`: Get the length of the chat history.

- `get_unique_chat_history_length() -> int`: Get the length of the unique chat history.

- `get_chat_history_tokens() -> int`: Get the total number of tokens in the chat history.

- `print_beautiful(content: str, color: str = 'cyan')`: Print content beautifully using colored text.

- `stream(content: str)`: Stream the content, printing it character by character.

- `meta_prompt() -> str`: Get the meta prompt that provides guidance for including image labels in responses.

## 5. Usage Examples <a name="usage-examples"></a>

Let's explore some usage examples of the MultiModalAI library:

### Example 1: Running

 the Model

```python
# Import the library
from swarms.models import BaseMultiModalModel

# Create an instance of the model
model = BaseMultiModalModel(
    model_name="your_model_name",
    temperature=0.5,
    max_tokens=500,
    device="cuda",
)

# Run the model with a text task and an image URL
response = model.run(
    "Generate a summary of this text", "https://www.example.com/image.jpg"
)
print(response)
```

### Example 2: Running Multiple Tasks Concurrently

```python
# Import the library
from swarms.models import BaseMultiModalModel

# Create an instance of the model
model = BaseMultiModalModel(
    model_name="your_model_name",
    temperature=0.5,
    max_tokens=500,
    max_workers=4,
    device="cuda",
)

# Define a list of tasks and image URLs
tasks = ["Task 1", "Task 2", "Task 3"]
images = ["https://image1.jpg", "https://image2.jpg", "https://image3.jpg"]

# Run the model on multiple tasks concurrently
responses = model.run_many(tasks, images)
for response in responses:
    print(response)
```

### Example 3: Running the Model Asynchronously

```python
# Import the library
from swarms.models import BaseMultiModalModel

# Create an instance of the model
model = BaseMultiModalModel(
    model_name="your_model_name",
    temperature=0.5,
    max_tokens=500,
    device="cuda",
)

# Define a list of tasks and image URLs
tasks_images = [
    ("Task 1", "https://image1.jpg"),
    ("Task 2", "https://image2.jpg"),
    ("Task 3", "https://image3.jpg"),
]

# Run the model on multiple tasks asynchronously
responses = model.run_batch_async(tasks_images)
for response in responses:
    print(response)
```

### Example 4: Inheriting `BaseMultiModalModel` for it's prebuilt classes
```python
from swarms.models import BaseMultiModalModel


class CustomMultiModalModel(BaseMultiModalModel):
    def __init__(self, model_name, custom_parameter, *args, **kwargs):
        # Call the parent class constructor
        super().__init__(model_name=model_name, *args, **kwargs)
        # Initialize custom parameters specific to your model
        self.custom_parameter = custom_parameter

    def __call__(self, text, img):
        # Implement the multimodal model logic here
        # You can use self.custom_parameter and other inherited attributes
        pass

    def generate_summary(self, text):
        # Implement the summary generation logic using your model
        # You can use self.custom_parameter and other inherited attributes
        pass


# Create an instance of your custom multimodal model
custom_model = CustomMultiModalModel(
    model_name="your_custom_model_name",
    custom_parameter="your_custom_value",
    temperature=0.5,
    max_tokens=500,
    device="cuda",
)

# Run your custom model
response = custom_model.run(
    "Generate a summary of this text", "https://www.example.com/image.jpg"
)
print(response)

# Generate a summary using your custom model
summary = custom_model.generate_summary("This is a sample text to summarize.")
print(summary)
```

In the code above:

1. We define a `CustomMultiModalModel` class that inherits from `BaseMultiModalModel`.

2. In the constructor of our custom class, we call the parent class constructor using `super()` and initialize any custom parameters specific to our model. In this example, we introduced a `custom_parameter`.

3. We override the `__call__` method, which is responsible for running the multimodal model logic. Here, you can implement the specific behavior of your model, considering both text and image inputs.

4. We override the `generate_summary` method, which is used to generate a summary of text input. You can implement your custom summarization logic here.

5. We create an instance of our custom model, passing the required parameters, including the custom parameter.

6. We demonstrate how to run the custom model and generate a summary using it.

By inheriting from `BaseMultiModalModel`, you can leverage the prebuilt features and methods provided by the library while customizing the behavior of your multimodal model. This allows you to create powerful and specialized models for various multimodal tasks.

These examples demonstrate how to use MultiModalAI to run multimodal models with text and image inputs. You can adjust the parameters and methods to suit your specific use cases.

## 6. Additional Tips <a name="additional-tips"></a>

Here are some additional tips and considerations for using MultiModalAI effectively:

- **Custom Models**: You can create your own multimodal models and inherit from the `BaseMultiModalModel` class to integrate them with this library.

- **Retries**: In cases where text generation might fail due to various reasons (e.g., server issues), using methods with retries can be helpful.

- **Monitoring**: You can monitor the performance of your model using methods like `_tokens_per_second()` and `_time_for_generation()`.

- **Chat History**: The library maintains a chat history, allowing you to keep track of interactions.

- **Streaming**: The `stream()` method can be useful for displaying output character by character, which can be helpful for certain applications.

## 7. References and Resources <a name="references-and-resources"></a>

Here are some references and resources that you may find useful for working with multimodal models:

- [Hugging Face Transformers Library](https://huggingface.co/transformers/): A library for working with various transformer-based models.

- [PIL (Python Imaging Library)](https://pillow.readthedocs.io/en/stable/): Documentation for working with images in Python using the Pillow library.

- [Concurrent Programming in Python](https://docs.python.org/3/library/concurrent.futures.html): Official Python documentation for concurrent programming.

- [Requests Library Documentation](https://docs.python-requests.org/en/latest/): Documentation for the Requests library, which is used for making HTTP requests.

- [Base64 Encoding in Python](https://docs.python.org/3/library/base64.html): Official Python documentation for base64 encoding and decoding.

This concludes the documentation for the MultiModalAI library. You can now explore the library further and integrate it with your multimodal AI projects.