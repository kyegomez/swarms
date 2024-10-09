import logging
from typing import Any, Dict, Optional

import requests
from pydantic import BaseModel, Field
from swarms import Conversation
from swarm_models import OpenAIFunctionCaller
from loguru import logger
import os


class APITaskSchema(BaseModel):
    plan: str = Field(
        ...,
        description="Plan out the API request to be executed, contemplate the endpoint, method, headers, body, and params.",
    )
    url: str = Field(
        ..., description="The API endpoint to send the request to."
    )
    method: str = Field(
        ...,
        description="HTTP method to use for the request (e.g., GET, POST).",
    )
    headers: Optional[Dict[str, str]] = Field(
        ..., description="Optional headers to include in the request."
    )
    body: Optional[Dict[str, Any]] = Field(
        ..., description="Optional body content for POST requests."
    )
    params: Optional[Dict[str, Any]] = Field(
        ..., description="Optional query parameters for the request."
    )


class APIRequestAgent:
    """
    An agent that sends API requests based on user input.

    Args:
        name (str, optional): The name of the agent. Defaults to "APIRequestAgent".
        description (str, optional): The description of the agent. Defaults to "An agent that sends API requests based on user input.".
        schema (BaseModel, optional): The schema for the API task. Defaults to APITaskSchema.
        temperature (int, optional): The temperature for the language model. Defaults to 0.5.
        system_prompt (str, optional): The system prompt for the language model. Defaults to "You are an API request manager. Create and execute requests based on the user's needs.".
        max_tokens (int, optional): The maximum number of tokens for the language model. Defaults to 4000.
        full_agent_history (str, optional): The full agent history. Defaults to None.
        max_loops (int, optional): The maximum number of loops for the agent. Defaults to 10.

    Attributes:
        name (str): The name of the agent.
        description (str): The description of the agent.
        schema (BaseModel): The schema for the API task.
        session (requests.Session): The session for connection pooling.
        system_prompt (str): The system prompt for the language model.
        max_tokens (int): The maximum number of tokens for the language model.
        full_agent_history (str): The full agent history.
        max_loops (int): The maximum number of loops for the agent.
        llm (OpenAIFunctionCaller): The function caller for the language model.
        conversation (Conversation): The conversation object.
    """

    def __init__(
        self,
        name: str = "APIRequestAgent",
        description: str = "An agent that sends API requests based on user input.",
        schema: BaseModel = APITaskSchema,
        temperature: int = 0.5,
        system_prompt: str = "You are an API request manager. Create and execute requests based on the user's needs.",
        max_tokens: int = 4000,
        full_agent_history: str = None,
        max_loops: int = 10,
        *args,
        **kwargs,
    ):
        # super().__init__(name=name, *args, **kwargs)
        self.name = name
        self.description = description
        self.schema = schema
        self.session = (
            requests.Session()
        )  # Optional: Use a session for connection pooling.
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.full_agent_history = full_agent_history
        self.max_loops = max_loops

        # Initialize the function caller (LLM) with the schema
        self.llm = OpenAIFunctionCaller(
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            base_model=APITaskSchema,
            parallel_tool_calls=False,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

        # Conversation
        self.conversation = Conversation(
            time_enabled=True,
            system_prompt=system_prompt,
        )

        # Full Agent history
        self.full_agent_history = (
            self.conversation.return_history_as_string()
        )

    def parse_response(
        self, response: requests.Response
    ) -> Dict[str, Any]:
        """
        Parses the API response and returns the content.

        Args:
            response (requests.Response): The API response to parse.

        Returns:
            Dict[str, Any]: The parsed response content.
        """
        try:
            logger.info(
                f"Response status code: {response.status_code}"
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTPError: {e}")
            raise
        except ValueError as e:
            logging.error(f"Failed to parse JSON: {e}")
            raise

    def execute_request(self, task: APITaskSchema) -> Dict[str, Any]:
        """
        Executes the API request based on the given task schema.

        Args:
            task (APITaskSchema): The task schema containing request details.

        Returns:
            Dict[str, Any]: The API response.
        """
        base_url = task.url

        url = f"{base_url}/{task.endpoint}"
        method = task.method.upper()

        logger.info(f"Executing request: {method} {url}")
        try:
            if method == "GET":
                response = self.session.get(
                    url, headers=task.headers, params=task.params
                )
            elif method == "POST":
                response = self.session.post(
                    url,
                    headers=task.headers,
                    json=task.body,
                    params=task.params,
                )
            elif method == "PUT":
                response = self.session.put(
                    url,
                    headers=task.headers,
                    json=task.body,
                    params=task.params,
                )
            elif method == "DELETE":
                response = self.session.delete(
                    url, headers=task.headers, params=task.params
                )

            elif method == "PATCH":
                response = self.session.patch(
                    url,
                    headers=task.headers,
                    json=task.body,
                    params=task.params,
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            logging.info(f"Executed {method} request to {url}")
            return self.parse_response(response)

        except requests.exceptions.RequestException as e:
            logging.error(f"RequestException: {e}")
            raise

    def execute_api_request(
        self, task: APITaskSchema
    ) -> Dict[str, Any]:
        """
        Executes a single step: sends the request and processes the response.

        Args:
            task (APITaskSchema): The task schema containing request details.

        Returns:
            Dict[str, Any]: The processed response from the API.
        """
        logger.info(f"Executing API request based on task: {task}")
        response = self.execute_request(task)
        response = str(response)

        # Log the response in the conversation
        self.conversation.add(role="API", content=response)
        return response

    def run(self, task: str) -> Any:
        """
        Runs the agent by processing a task string, and executing the requests.

        Args:
            task (str): The task to be processed by the LLM and executed by the agent.

        Returns:
            Any: The result of the task processed by the LLM.
        """
        logger.info(f"Running agent with task: {task}")
        output = self.llm.run(task)

        # Log the output in the conversation
        print(output)
        print(type(output))
        self.conversation.add(role=self.name, content=output)

        # Convert dict -> APITaskSchema
        output = APITaskSchema(**output)

        logger.info(f"Executing request based on task: {output}")
        return self.execute_api_request(output)


# Model
agent = APIRequestAgent(
    name="APIRequestAgent",
    description="An agent that sends API requests based on user input.",
    schema=APITaskSchema,
    system_prompt="You are an API request manager. Create and execute requests based on the user's needs.",
)

agent.run("Send an API request to an open source API")

print(agent.full_agent_history)
