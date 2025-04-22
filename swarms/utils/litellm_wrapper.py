import base64
import requests

import asyncio
from typing import List

from loguru import logger
import litellm

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


class LiteLLMException(Exception):
    """
    Exception for LiteLLM.
    """


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
        retries: int = 3,
        verbose: bool = False,
        caching: bool = False,
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
        self.caching = caching
        self.modalities = []
        self._cached_messages = {}  # Cache for prepared messages
        self.messages = []  # Initialize messages list

        # Configure litellm settings
        litellm.set_verbose = (
            verbose  # Disable verbose mode for better performance
        )
        litellm.ssl_verify = ssl_verify
        litellm.num_retries = (
            retries  # Add retries for better reliability
        )

    def _prepare_messages(self, task: str) -> list:
        """
        Prepare the messages for the given task.

        Args:
            task (str): The task to prepare messages for.

        Returns:
            list: A list of messages prepared for the task.
        """
        # Check cache first
        cache_key = f"{self.system_prompt}:{task}"
        if cache_key in self._cached_messages:
            return self._cached_messages[cache_key].copy()

        messages = []
        if self.system_prompt:
            messages.append(
                {"role": "system", "content": self.system_prompt}
            )
        messages.append({"role": "user", "content": task})

        # Cache the prepared messages
        self._cached_messages[cache_key] = messages.copy()
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
        self.messages = []  # Reset messages
        self.modalities.append("text")

        if audio is not None:
            self.audio_processing(task=task, audio=audio)
            self.modalities.append("audio")

        if img is not None:
            self.vision_processing(task=task, image=img)
            self.modalities.append("vision")

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
            audio (str, optional): Audio input if any. Defaults to None.
            img (str, optional): Image input if any. Defaults to None.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The content of the response from the model.

        Raises:
            Exception: If there is an error in processing the request.
        """
        try:
            messages = self._prepare_messages(task)

            if audio is not None or img is not None:
                self.handle_modalities(
                    task=task, audio=audio, img=img
                )
                messages = (
                    self.messages
                )  # Use modality-processed messages

            if (
                self.model_name == "openai/o4-mini"
                or self.model_name == "openai/o3-2025-04-16"
            ):
                # Prepare common completion parameters
                completion_params = {
                    "model": self.model_name,
                    "messages": messages,
                    "stream": self.stream,
                    # "temperature": self.temperature,
                    "max_completion_tokens": self.max_tokens,
                    "caching": self.caching,
                    **kwargs,
                }

            else:
                # Prepare common completion parameters
                completion_params = {
                    "model": self.model_name,
                    "messages": messages,
                    "stream": self.stream,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "caching": self.caching,
                    **kwargs,
                }

            # Handle tool-based completion
            if self.tools_list_dictionary is not None:
                completion_params.update(
                    {
                        "tools": self.tools_list_dictionary,
                        "tool_choice": self.tool_choice,
                        "parallel_tool_calls": self.parallel_tool_calls,
                    }
                )
                response = completion(**completion_params)
                return (
                    response.choices[0]
                    .message.tool_calls[0]
                    .function.arguments
                )

            # Handle modality-based completion
            if (
                self.modalities and len(self.modalities) > 1
            ):  # More than just text
                completion_params.update(
                    {"modalities": self.modalities}
                )
                response = completion(**completion_params)
                return response.choices[0].message.content

            # Standard completion
            if self.stream:
                return completion(**completion_params)
            else:
                response = completion(**completion_params)
                return response.choices[0].message.content

        except LiteLLMException as error:
            logger.error(f"Error in LiteLLM run: {str(error)}")
            if "rate_limit" in str(error).lower():
                logger.warning(
                    "Rate limit hit, retrying with exponential backoff..."
                )
                import time

                time.sleep(2)  # Add a small delay before retry
                return self.run(task, audio, img, *args, **kwargs)
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
        Run the LLM model asynchronously for the given task.

        Args:
            task (str): The task to run the model for.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The content of the response from the model.
        """
        try:
            messages = self._prepare_messages(task)

            # Prepare common completion parameters
            completion_params = {
                "model": self.model_name,
                "messages": messages,
                "stream": self.stream,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **kwargs,
            }

            # Handle tool-based completion
            if self.tools_list_dictionary is not None:
                completion_params.update(
                    {
                        "tools": self.tools_list_dictionary,
                        "tool_choice": self.tool_choice,
                        "parallel_tool_calls": self.parallel_tool_calls,
                    }
                )
                response = await acompletion(**completion_params)
                return (
                    response.choices[0]
                    .message.tool_calls[0]
                    .function.arguments
                )

            # Standard completion
            response = await acompletion(**completion_params)

            print(response)
            return response

        except Exception as error:
            logger.error(f"Error in LiteLLM arun: {str(error)}")
            # if "rate_limit" in str(error).lower():
            #     logger.warning(
            #         "Rate limit hit, retrying with exponential backoff..."
            #     )
            #     await asyncio.sleep(2)  # Use async sleep
            #     return await self.arun(task, *args, **kwargs)
            raise error

    async def _process_batch(
        self, tasks: List[str], batch_size: int = 10
    ):
        """
        Process a batch of tasks asynchronously.

        Args:
            tasks (List[str]): List of tasks to process.
            batch_size (int): Size of each batch.

        Returns:
            List[str]: List of responses.
        """
        results = []
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            batch_results = await asyncio.gather(
                *[self.arun(task) for task in batch],
                return_exceptions=True,
            )

            # Handle any exceptions in the batch
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(
                        f"Error in batch processing: {str(result)}"
                    )
                    results.append(str(result))
                else:
                    results.append(result)

            # Add a small delay between batches to avoid rate limits
            if i + batch_size < len(tasks):
                await asyncio.sleep(0.5)

        return results

    def batched_run(self, tasks: List[str], batch_size: int = 10):
        """
        Run multiple tasks in batches synchronously.

        Args:
            tasks (List[str]): List of tasks to process.
            batch_size (int): Size of each batch.

        Returns:
            List[str]: List of responses.
        """
        logger.info(
            f"Running {len(tasks)} tasks in batches of {batch_size}"
        )
        return asyncio.run(self._process_batch(tasks, batch_size))

    async def batched_arun(
        self, tasks: List[str], batch_size: int = 10
    ):
        """
        Run multiple tasks in batches asynchronously.

        Args:
            tasks (List[str]): List of tasks to process.
            batch_size (int): Size of each batch.

        Returns:
            List[str]: List of responses.
        """
        logger.info(
            f"Running {len(tasks)} tasks asynchronously in batches of {batch_size}"
        )
        return await self._process_batch(tasks, batch_size)
