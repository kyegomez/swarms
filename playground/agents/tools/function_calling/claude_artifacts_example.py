from swarms.models.openai_function_caller import OpenAIFunctionCaller
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
