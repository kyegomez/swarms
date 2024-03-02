from typing import Any, Dict, List, Optional, Union

import openai
import requests
from pydantic import BaseModel, validator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from termcolor import colored


class FunctionSpecification(BaseModel):
    """
    Defines the specification for a function including its parameters and metadata.

    Attributes:
    -----------
    name: str
        The name of the function.
    description: str
        A brief description of what the function does.
    parameters: Dict[str, Any]
        The parameters required by the function, with their details.
    required: Optional[List[str]]
        List of required parameter names.

    Methods:
    --------
    validate_params(params: Dict[str, Any]) -> None:
        Validates the parameters against the function's specification.



    Example:

    # Example Usage
    def get_current_weather(location: str, format: str) -> str:
    ``'
    Example function to get current weather.

    Args:
        location (str): The city and state, e.g. San Francisco, CA.
        format (str): The temperature unit, e.g. celsius or fahrenheit.

    Returns:
        str: Weather information.
    '''
    # Implementation goes here
    return "Sunny, 23°C"


    weather_function_spec = FunctionSpecification(
        name="get_current_weather",
        description="Get the current weather",
        parameters={
            "location": {"type": "string", "description": "The city and state"},
            "format": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The temperature unit",
            },
        },
        required=["location", "format"],
    )

    # Validating parameters for the function
    params = {"location": "San Francisco, CA", "format": "celsius"}
    weather_function_spec.validate_params(params)

    # Calling the function
    print(get_current_weather(**params))
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    required: Optional[List[str]] = None

    @validator("parameters")
    def check_parameters(cls, params):
        if not isinstance(params, dict):
            raise ValueError("Parameters must be a dictionary.")
        return params

    def validate_params(self, params: Dict[str, Any]) -> None:
        """
        Validates the parameters against the function's specification.

        Args:
            params (Dict[str, Any]): The parameters to validate.

        Raises:
            ValueError: If any required parameter is missing or if any parameter is invalid.
        """
        for key, value in params.items():
            if key in self.parameters:
                self.parameters[key]
                # Perform specific validation based on param_spec
                # This can include type checking, range validation, etc.
            else:
                raise ValueError(f"Unexpected parameter: {key}")

        for req_param in self.required or []:
            if req_param not in params:
                raise ValueError(
                    f"Missing required parameter: {req_param}"
                )


class OpenAIFunctionCaller:
    def __init__(
        self,
        openai_api_key: str,
        model: str = "text-davinci-003",
        max_tokens: int = 3000,
        temperature: float = 0.5,
        top_p: float = 1.0,
        n: int = 1,
        stream: bool = False,
        stop: Optional[str] = None,
        echo: bool = False,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logprobs: Optional[int] = None,
        best_of: int = 1,
        logit_bias: Dict[str, float] = None,
        user: str = None,
        messages: List[Dict] = None,
        timeout_sec: Union[float, None] = None,
    ):
        self.openai_api_key = openai_api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.stream = stream
        self.stop = stop
        self.echo = echo
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.logprobs = logprobs
        self.best_of = best_of
        self.logit_bias = logit_bias
        self.user = user
        self.messages = messages if messages is not None else []
        self.timeout_sec = timeout_sec

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
    )
    def chat_completion_request(
        self,
        messages,
        tools=None,
        tool_choice=None,
    ):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + openai.api_key,
        }
        json_data = {"model": self.model, "messages": messages}
        if tools is not None:
            json_data.update({"tools": tools})
        if tool_choice is not None:
            json_data.update({"tool_choice": tool_choice})
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=json_data,
            )
            return response
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return e

    def pretty_print_conversation(self, messages):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "tool": "magenta",
        }

        for message in messages:
            if message["role"] == "system":
                print(
                    colored(
                        f"system: {message['content']}\n",
                        role_to_color[message["role"]],
                    )
                )
            elif message["role"] == "user":
                print(
                    colored(
                        f"user: {message['content']}\n",
                        role_to_color[message["role"]],
                    )
                )
            elif message["role"] == "assistant" and message.get(
                "function_call"
            ):
                print(
                    colored(
                        f"assistant: {message['function_call']}\n",
                        role_to_color[message["role"]],
                    )
                )
            elif message["role"] == "assistant" and not message.get(
                "function_call"
            ):
                print(
                    colored(
                        f"assistant: {message['content']}\n",
                        role_to_color[message["role"]],
                    )
                )
            elif message["role"] == "tool":
                print(
                    colored(
                        f"function ({message['name']}):"
                        f" {message['content']}\n",
                        role_to_color[message["role"]],
                    )
                )

    def call(self, task: str, *args, **kwargs) -> Dict:
        return openai.Completion.create(
            engine=self.model,
            prompt=task,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            n=self.n,
            stream=self.stream,
            stop=self.stop,
            echo=self.echo,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            logprobs=self.logprobs,
            best_of=self.best_of,
            logit_bias=self.logit_bias,
            user=self.user,
            messages=self.messages,
            timeout_sec=self.timeout_sec,
            *args,
            **kwargs,
        )

    def run(self, task: str, *args, **kwargs) -> str:
        response = self.call(task, *args, **kwargs)
        return response["choices"][0]["text"].strip()
