import asyncio
from typing import List

from loguru import logger


try:
    from litellm import completion, acompletion
except ImportError:
    import subprocess
    import sys
    import litellm

    print("Installing litellm")

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-U", "litellm"]
    )
    print("litellm installed")

    from litellm import completion

    litellm.set_verbose = True
    litellm.ssl_verify = False


class LiteLLM:
    """
    This class represents a LiteLLM.
    It is used to interact with the LLM model for various tasks.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        system_prompt: str = None,
        stream: bool = False,
        temperature: float = 0.5,
        max_tokens: int = 4000,
        ssl_verify: bool = False,
        max_completion_tokens: int = 4000,
        tools_list_dictionary: List[dict] = None,
        tool_choice: str = "auto",
        parallel_tool_calls: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize the LiteLLM with the given parameters.

        Args:
            model_name (str, optional): The name of the model to use. Defaults to "gpt-4o".
            system_prompt (str, optional): The system prompt to use. Defaults to None.
            stream (bool, optional): Whether to stream the output. Defaults to False.
            temperature (float, optional): The temperature for the model. Defaults to 0.5.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 4000.
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.stream = stream
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ssl_verify = ssl_verify
        self.max_completion_tokens = max_completion_tokens
        self.tools_list_dictionary = tools_list_dictionary
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls

    def _prepare_messages(self, task: str) -> list:
        """
        Prepare the messages for the given task.

        Args:
            task (str): The task to prepare messages for.

        Returns:
            list: A list of messages prepared for the task.
        """
        messages = []

        if self.system_prompt:  # Check if system_prompt is not None
            messages.append(
                {"role": "system", "content": self.system_prompt}
            )

        messages.append({"role": "user", "content": task})

        return messages

    def run(self, task: str, *args, **kwargs):
        """
        Run the LLM model for the given task.

        Args:
            task (str): The task to run the model for.
            *args: Additional positional arguments to pass to the model.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            str: The content of the response from the model.
        """
        try:

            messages = self._prepare_messages(task)

            if self.tools_list_dictionary is not None:
                response = completion(
                    model=self.model_name,
                    messages=messages,
                    stream=self.stream,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tools=self.tools_list_dictionary,
                    tool_choice=self.tool_choice,
                    parallel_tool_calls=self.parallel_tool_calls,
                    *args,
                    **kwargs,
                )

                return (
                    response.choices[0]
                    .message.tool_calls[0]
                    .function.arguments
                )

            else:
                response = completion(
                    model=self.model_name,
                    messages=messages,
                    stream=self.stream,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    *args,
                    **kwargs,
                )

                content = response.choices[
                    0
                ].message.content  # Accessing the content

                return content
        except Exception as error:
            logger.error(f"Error in LiteLLM: {error}")
            raise error

    def __call__(self, task: str, *args, **kwargs):
        """
        Call the LLM model for the given task.

        Args:
            task (str): The task to run the model for.
            *args: Additional positional arguments to pass to the model.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            str: The content of the response from the model.
        """
        return self.run(task, *args, **kwargs)

    async def arun(self, task: str, *args, **kwargs):
        """
        Run the LLM model for the given task.

        Args:
            task (str): The task to run the model for.
            *args: Additional positional arguments to pass to the model.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            str: The content of the response from the model.
        """
        try:
            messages = self._prepare_messages(task)

            if self.tools_list_dictionary is not None:
                response = await acompletion(
                    model=self.model_name,
                    messages=messages,
                    stream=self.stream,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tools=self.tools_list_dictionary,
                    tool_choice=self.tool_choice,
                    parallel_tool_calls=self.parallel_tool_calls,
                    *args,
                    **kwargs,
                )

                content = (
                    response.choices[0]
                    .message.tool_calls[0]
                    .function.arguments
                )

                # return response

            else:
                response = await acompletion(
                    model=self.model_name,
                    messages=messages,
                    stream=self.stream,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    *args,
                    **kwargs,
                )

                content = response.choices[
                    0
                ].message.content  # Accessing the content

            return content
        except Exception as error:
            logger.error(f"Error in LiteLLM: {error}")
            raise error

    def batched_run(self, tasks: List[str], batch_size: int = 10):
        """
        Run the LLM model for the given tasks in batches.
        """
        logger.info(
            f"Running tasks in batches of size {batch_size}. Total tasks: {len(tasks)}"
        )
        results = []
        for task in tasks:
            logger.info(f"Running task: {task}")
            results.append(self.run(task))
        logger.info("Completed all tasks.")
        return results

    def batched_arun(self, tasks: List[str], batch_size: int = 10):
        """
        Run the LLM model for the given tasks in batches.
        """
        logger.info(
            f"Running asynchronous tasks in batches of size {batch_size}. Total tasks: {len(tasks)}"
        )
        results = []
        for task in tasks:
            logger.info(f"Running asynchronous task: {task}")
            results.append(asyncio.run(self.arun(task)))
        logger.info("Completed all asynchronous tasks.")
        return results
