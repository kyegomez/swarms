import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import List

from loguru import logger
from pydantic import BaseModel


try:
    from openai import OpenAI
except ImportError:
    logger.error(
        "OpenAI library not found. Please install the OpenAI library by running 'pip install openai'"
    )
    import sys

    subprocess.run([sys.executable, "-m", "pip", "install", "openai"])
    from openai import OpenAI


SUPPORTED_MODELS = [
    "o3-mini-2025-1-31",
    "o1-2024-12-17",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-08-06",
]


def check_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError(
            "API key is not set. Please set the API key using the api_key parameter."
        )
    return api_key


class OpenAIFunctionCaller:
    """
    A class to interact with the OpenAI API for generating text based on a system prompt and a task.

    Attributes:
    - system_prompt (str): The system prompt to guide the AI's response.
    - api_key (str): The API key for the OpenAI service.
    - temperature (float): The temperature parameter for the AI model, controlling randomness.
    - base_model (BaseModel): The Pydantic model to parse the response into.
    - max_tokens (int): The maximum number of tokens in the response.
    - client (OpenAI): The OpenAI client instance.
    """

    def __init__(
        self,
        system_prompt: str,
        base_model: BaseModel,
        api_key: str = os.getenv("OPENAI_API_KEY"),
        temperature: float = 0.1,
        max_tokens: int = 5000,
        model_name: str = "gpt-4o-2024-08-06",
    ):
        self.system_prompt = system_prompt
        self.api_key = api_key
        self.temperature = temperature
        self.base_model = base_model
        self.max_tokens = max_tokens
        self.model_name = model_name

        self.client = OpenAI(api_key=self.api_key)

    def run(self, task: str):
        """
        Run the OpenAI model with the system prompt and task to generate a response.

        Args:
        - task (str): The task to be completed.
        - *args: Additional positional arguments for the OpenAI API.
        - **kwargs: Additional keyword arguments for the OpenAI API.

        Returns:
        - BaseModel: The parsed response based on the base_model.
        """
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": task},
                ],
                response_format=self.base_model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            return completion.choices[0].message.parsed

        except Exception as e:
            print(f"There was an error: {e}")

    def check_model_support(self):
        # need to print the supported models
        for model in SUPPORTED_MODELS:
            print(model)

        return SUPPORTED_MODELS

    def batch_run(self, tasks: List[str]) -> List[BaseModel]:
        """
        Batch run the OpenAI model with the system prompt and task to generate a response.
        """
        return [self.run(task) for task in tasks]

    def concurrent_run(self, tasks: List[str]) -> List[BaseModel]:
        """
        Concurrent run the OpenAI model with the system prompt and task to generate a response.
        """
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            return list(executor.map(self.run, tasks))


# class TestModel(BaseModel):
#     name: str
#     age: int

# # Example usage
# model = OpenAIFunctionCaller(
#     system_prompt="You are a helpful assistant that returns structured data about people.",
#     base_model=TestModel,
#     api_key=os.getenv("OPENAI_API_KEY"),
#     temperature=0.7,
#     max_tokens=1000
# )

# # Test with a more appropriate prompt for the TestModel schema
# response = model.run("Tell me about a person named John who is 25 years old")
# print(response)
