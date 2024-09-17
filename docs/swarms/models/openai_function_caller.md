# OpenAIFunctionCaller Documentation

The `OpenAIFunctionCaller` class is designed to interface with OpenAI's chat completion API, allowing users to generate responses based on given prompts using specified models. This class encapsulates the setup and execution of API calls, including handling API keys, model parameters, and response formatting. The class extends the `BaseLLM` and utilizes OpenAI's client library to facilitate interactions.

## Class Definition

### OpenAIFunctionCaller

A class that represents a caller for OpenAI chat completions.

### Attributes

| Attribute            | Type              | Description                                                             |
|----------------------|-------------------|-------------------------------------------------------------------------|
| `system_prompt`      | `str`             | The system prompt to be used in the chat completion.                    |
| `model_name`         | `str`             | The name of the OpenAI model to be used.                                |
| `max_tokens`         | `int`             | The maximum number of tokens in the generated completion.               |
| `temperature`        | `float`           | The temperature parameter for randomness in the completion.             |
| `base_model`         | `BaseModel`       | The base model to be used for the completion.                           |
| `parallel_tool_calls`| `bool`            | Whether to make parallel tool calls.                                    |
| `top_p`              | `float`           | The top-p parameter for nucleus sampling in the completion.             |
| `client`             | `openai.OpenAI`   | The OpenAI client for making API calls.                                 |

### Methods

#### `check_api_key`

Checks if the API key is provided and retrieves it from the environment if not.

| Parameter     | Type   | Description                          |
|---------------|--------|--------------------------------------|
| None          |        |                                      |

**Returns:**

| Type   | Description                          |
|--------|--------------------------------------|
| `str`  | The API key.                         |

#### `run`

Runs the chat completion with the given task and returns the generated completion.

| Parameter | Type     | Description                                                     |
|-----------|----------|-----------------------------------------------------------------|
| `task`    | `str`    | The user's task for the chat completion.                        |
| `*args`   |          | Additional positional arguments to be passed to the OpenAI API. |
| `**kwargs`|          | Additional keyword arguments to be passed to the OpenAI API.    |

**Returns:**

| Type   | Description                                   |
|--------|-----------------------------------------------|
| `str`  | The generated completion.                     |

#### `convert_to_dict_from_base_model`

Converts a `BaseModel` to a dictionary.

| Parameter   | Type       | Description                          |
|-------------|------------|--------------------------------------|
| `base_model`| `BaseModel`| The BaseModel to be converted.       |

**Returns:**

| Type   | Description                          |
|--------|--------------------------------------|
| `dict` | A dictionary representing the BaseModel.|

#### `convert_list_of_base_models`

Converts a list of `BaseModels` to a list of dictionaries.

| Parameter    | Type            | Description                          |
|--------------|-----------------|--------------------------------------|
| `base_models`| `List[BaseModel]`| A list of BaseModels to be converted.|

**Returns:**

| Type   | Description                                   |
|--------|-----------------------------------------------|
| `List[Dict]` | A list of dictionaries representing the converted BaseModels.  |

## Usage Examples

Here are three examples demonstrating different ways to use the `OpenAIFunctionCaller` class:

### Example 1: Production-Grade Claude Artifacts

```python
import openai
from swarm_models.openai_function_caller import OpenAIFunctionCaller
from swarms.artifacts.main_artifact import Artifact


# Pydantic is a data validation library that provides data validation and parsing using Python type hints.


# Example usage:
# Initialize the function caller
model = OpenAIFunctionCaller(
    system_prompt="You're a helpful assistant.The time is August 6, 2024",
    max_tokens=500,
    temperature=0.5,
    base_model=Artifact,
    parallel_tool_calls=False,
)


# The OpenAIFunctionCaller class is used to interact with the OpenAI API and make function calls.
# Here, we initialize an instance of the OpenAIFunctionCaller class with the following parameters:
# - system_prompt: A prompt that sets the context for the conversation with the API.
# - max_tokens: The maximum number of tokens to generate in the API response.
# - temperature: A parameter that controls the randomness of the generated text.
# - base_model: The base model to use for the API calls, in this case, the WeatherAPI class.
out = model.run("Create a python file with a python game code in it")
print(out)
```

### Example 2: Prompt Generator

```python
from swarm_models.openai_function_caller import OpenAIFunctionCaller
from pydantic import BaseModel, Field
from typing import Sequence


class PromptUseCase(BaseModel):
    use_case_name: str = Field(
        ...,
        description="The name of the use case",
    )
    use_case_description: str = Field(
        ...,
        description="The description of the use case",
    )


class PromptSpec(BaseModel):
    prompt_name: str = Field(
        ...,
        description="The name of the prompt",
    )
    prompt_description: str = Field(
        ...,
        description="The description of the prompt",
    )
    prompt: str = Field(
        ...,
        description="The prompt for the agent",
    )
    tags: str = Field(
        ...,
        description="The tags for the prompt such as sentiment, code, etc seperated by commas.",
    )
    use_cases: Sequence[PromptUseCase] = Field(
        ...,
        description="The use cases for the prompt",
    )


# Example usage:
# Initialize the function caller
model = OpenAIFunctionCaller(
    system_prompt="You're an agent creator, you're purpose is to create system prompt for new LLM Agents for the user. Follow the best practices for creating a prompt such as making it direct and clear. Providing instructions and many-shot examples will help the agent understand the task better.",
    max_tokens=1000,
    temperature=0.5,
    base_model=PromptSpec,
    parallel_tool_calls=False,
)


# The OpenAIFunctionCaller class is used to interact with the OpenAI API and make function calls.
out = model.run(
    "Create an prompt for generating quality rust code with instructions and examples."
)
print(out)

```

### Example 3: Sentiment Analysis 

```python
from swarm_models.openai_function_caller import OpenAIFunctionCaller
from pydantic import BaseModel, Field


# Pydantic is a data validation library that provides data validation and parsing using Python type hints.
# It is used here to define the data structure for making API calls to retrieve weather information.
class SentimentAnalysisCard(BaseModel):
    text: str = Field(
        ...,
        description="The text to be analyzed for sentiment rating",
    )
    rating: str = Field(
        ...,
        description="The sentiment rating of the text from 0.0 to 1.0",
    )


# The WeatherAPI class is a Pydantic BaseModel that represents the data structure
# for making API calls to retrieve weather information. It has two attributes: city and date.

# Example usage:
# Initialize the function caller
model = OpenAIFunctionCaller(
    system_prompt="You're a sentiment Analysis Agent, you're purpose is to rate the sentiment of text",
    max_tokens=100,
    temperature=0.5,
    base_model=SentimentAnalysisCard,
    parallel_tool_calls=False,
)


# The OpenAIFunctionCaller class is used to interact with the OpenAI API and make function calls.
# Here, we initialize an instance of the OpenAIFunctionCaller class with the following parameters:
# - system_prompt: A prompt that sets the context for the conversation with the API.
# - max_tokens: The maximum number of tokens to generate in the API response.
# - temperature: A parameter that controls the randomness of the generated text.
# - base_model: The base model to use for the API calls, in this case, the WeatherAPI class.
out = model.run("The hotel was average, but the food was excellent.")
print(out)

```

## Additional Information and Tips

- Ensure that your OpenAI API key is securely stored and not hard-coded into your source code. Use environment variables to manage sensitive information.
- Adjust the `temperature` and `top_p` parameters to control the randomness and diversity of the generated responses. Lower values for `temperature` will result in more deterministic outputs, while higher values will introduce more variability.
- When using `parallel_tool_calls`, ensure that the tools you are calling in parallel are thread-safe and can handle concurrent execution.

## References and Resources

- [OpenAI API Documentation](https://beta.openai.com/docs/)
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)
- [Loguru Logger Documentation](https://loguru.readthedocs.io/)

By following this comprehensive guide, you can effectively utilize the `OpenAIFunctionCaller` class to generate chat completions using OpenAI's models, customize the response parameters, and handle API interactions seamlessly within your application.