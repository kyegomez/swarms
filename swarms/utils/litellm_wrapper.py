import base64
import requests

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


def get_audio_base64(audio_source: str) -> str:
    """
    Convert audio from a given source to a base64 encoded string.

    This function handles both URLs and local file paths. If the audio source is a URL, it fetches the audio data
    from the internet. If it is a local file path, it reads the audio data from the specified file.

    Args:
        audio_source (str): The source of the audio, which can be a URL or a local file path.

    Returns:
        str: A base64 encoded string representation of the audio data.

    Raises:
        requests.HTTPError: If the HTTP request to fetch audio data fails.
        FileNotFoundError: If the local audio file does not exist.
    """
    # Handle URL
    if audio_source.startswith(("http://", "https://")):
        response = requests.get(audio_source)
        response.raise_for_status()
        audio_data = response.content
    # Handle local file
    else:
        with open(audio_source, "rb") as file:
            audio_data = file.read()

    encoded_string = base64.b64encode(audio_data).decode("utf-8")
    return encoded_string


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
        audio: str = None,
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
        self.modalities = ["text"]

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

    def audio_processing(self, task: str, audio: str):
        """
        Process the audio for the given task.

        Args:
            task (str): The task to be processed.
            audio (str): The path or identifier for the audio file.
        """
        self.modalities.append("audio")

        encoded_string = get_audio_base64(audio)

        # Append messages
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded_string,
                            "format": "wav",
                        },
                    },
                ],
            }
        )

    def vision_processing(self, task: str, image: str):
        """
        Process the image for the given task.
        """
        self.modalities.append("vision")

        # Append messages
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image,
                            # "detail": "high"
                            # "format": "image",
                        },
                    },
                ],
            }
        )

    def handle_modalities(
        self, task: str, audio: str = None, img: str = None
    ):
        """
        Handle the modalities for the given task.
        """
        if audio is not None:
            self.audio_processing(task=task, audio=audio)

        if img is not None:
            self.vision_processing(task=task, image=img)

        if audio is not None and img is not None:
            self.audio_processing(task=task, audio=audio)
            self.vision_processing(task=task, image=img)

    def run(
        self,
        task: str,
        audio: str = None,
        img: str = None,
        *args,
        **kwargs,
    ):
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

            self.handle_modalities(task=task, audio=audio, img=img)

            if self.tools_list_dictionary is not None:
                response = completion(
                    model=self.model_name,
                    messages=messages,
                    stream=self.stream,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tools=self.tools_list_dictionary,
                    modalities=self.modalities,
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
                    modalities=self.modalities,
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
