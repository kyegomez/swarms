import traceback
from typing import Optional
import base64
import requests
from pathlib import Path

import asyncio
from typing import List

from loguru import logger
import litellm
from pydantic import BaseModel

from litellm import completion, acompletion, supports_vision


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


def get_image_base64(image_source: str) -> str:
    """
    Convert image from a given source to a base64 encoded string.
    Handles URLs, local file paths, and data URIs.
    """
    # If already a data URI, return as is
    if image_source.startswith("data:image"):
        return image_source

    # Handle URL
    if image_source.startswith(("http://", "https://")):
        response = requests.get(image_source)
        response.raise_for_status()
        image_data = response.content
    # Handle local file
    else:
        with open(image_source, "rb") as file:
            image_data = file.read()

    # Get file extension for mime type
    extension = Path(image_source).suffix.lower()
    mime_type = (
        f"image/{extension[1:]}" if extension else "image/jpeg"
    )

    encoded_string = base64.b64encode(image_data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_string}"


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
        retries: int = 0,
        verbose: bool = False,
        caching: bool = False,
        mcp_call: bool = False,
        top_p: float = 1.0,
        functions: List[dict] = None,
        return_all: bool = False,
        base_url: str = None,
        api_key: str = None,
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
        self.mcp_call = mcp_call
        self.top_p = top_p
        self.functions = functions
        self.audio = audio
        self.return_all = return_all
        self.base_url = base_url
        self.api_key = api_key
        self.modalities = []
        self.messages = []  # Initialize messages list

        # Configure litellm settings
        litellm.set_verbose = (
            verbose  # Disable verbose mode for better performance
        )
        litellm.ssl_verify = ssl_verify
        litellm.num_retries = (
            retries  # Add retries for better reliability
        )

        litellm.drop_params = True

    def output_for_tools(self, response: any):
        if self.mcp_call is True:
            out = response.choices[0].message.tool_calls[0].function
            output = {
                "function": {
                    "name": out.name,
                    "arguments": out.arguments,
                }
            }
            return output
        else:
            out = response.choices[0].message.tool_calls

            if isinstance(out, BaseModel):
                out = out.model_dump()
            return out

    def _prepare_messages(
        self,
        task: str,
        img: str = None,
    ):
        """
        Prepare the messages for the given task.

        Args:
            task (str): The task to prepare messages for.

        Returns:
            list: A list of messages prepared for the task.
        """
        self.check_if_model_supports_vision(img=img)

        # Initialize messages
        messages = []

        # Add system prompt if present
        if self.system_prompt is not None:
            messages.append(
                {"role": "system", "content": self.system_prompt}
            )

        # Handle vision case
        if img is not None:
            messages = self.vision_processing(
                task=task, image=img, messages=messages
            )
        else:
            messages.append({"role": "user", "content": task})

        return messages

    def anthropic_vision_processing(
        self, task: str, image: str, messages: list
    ) -> list:
        """
        Process vision input specifically for Anthropic models.
        Handles Anthropic's specific image format requirements.
        """
        # Get base64 encoded image
        image_url = get_image_base64(image)

        # Extract mime type from the data URI or use default
        mime_type = "image/jpeg"  # default
        if "data:" in image_url and ";base64," in image_url:
            mime_type = image_url.split(";base64,")[0].split("data:")[
                1
            ]

        # Ensure mime type is one of the supported formats
        supported_formats = [
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
        ]
        if mime_type not in supported_formats:
            mime_type = (
                "image/jpeg"  # fallback to jpeg if unsupported
            )

        # Construct Anthropic vision message
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "format": mime_type,
                        },
                    },
                ],
            }
        )

        return messages

    def openai_vision_processing(
        self, task: str, image: str, messages: list
    ) -> list:
        """
        Process vision input specifically for OpenAI models.
        Handles OpenAI's specific image format requirements.
        """
        # Get base64 encoded image with proper format
        image_url = get_image_base64(image)

        # Prepare vision message
        vision_message = {
            "type": "image_url",
            "image_url": {"url": image_url},
        }

        # Add format for specific models
        extension = Path(image).suffix.lower()
        mime_type = (
            f"image/{extension[1:]}" if extension else "image/jpeg"
        )
        vision_message["image_url"]["format"] = mime_type

        # Append vision message
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task},
                    vision_message,
                ],
            }
        )

        return messages

    def vision_processing(
        self, task: str, image: str, messages: Optional[list] = None
    ):
        """
        Process the image for the given task.
        Handles different image formats and model requirements.
        """
        # # # Handle Anthropic models separately
        # # if "anthropic" in self.model_name.lower() or "claude" in self.model_name.lower():
        # #     messages = self.anthropic_vision_processing(task, image, messages)
        # #     return messages

        # # Get base64 encoded image with proper format
        # image_url = get_image_base64(image)

        # # Prepare vision message
        # vision_message = {
        #     "type": "image_url",
        #     "image_url": {"url": image_url},
        # }

        # # Add format for specific models
        # extension = Path(image).suffix.lower()
        # mime_type = f"image/{extension[1:]}" if extension else "image/jpeg"
        # vision_message["image_url"]["format"] = mime_type

        # # Append vision message
        # messages.append(
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "text", "text": task},
        #             vision_message,
        #         ],
        #     }
        # )

        # return messages
        if (
            "anthropic" in self.model_name.lower()
            or "claude" in self.model_name.lower()
        ):
            messages = self.anthropic_vision_processing(
                task, image, messages
            )
            return messages
        else:
            messages = self.openai_vision_processing(
                task, image, messages
            )
            return messages

    def audio_processing(self, task: str, audio: str):
        """
        Process the audio for the given task.

        Args:
            task (str): The task to be processed.
            audio (str): The path or identifier for the audio file.
        """
        encoded_string = get_audio_base64(audio)

        # Append audio message
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

    def check_if_model_supports_vision(self, img: str = None):
        """
        Check if the model supports vision.
        """
        if img is not None:
            out = supports_vision(model=self.model_name)

            if out is False:
                raise ValueError(
                    f"Model {self.model_name} does not support vision"
                )

    def run(
        self,
        task: str,
        audio: Optional[str] = None,
        img: Optional[str] = None,
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
            messages = self._prepare_messages(task=task, img=img)

            # Base completion parameters
            completion_params = {
                "model": self.model_name,
                "messages": messages,
                "stream": self.stream,
                "max_tokens": self.max_tokens,
                "caching": self.caching,
                "temperature": self.temperature,
                "top_p": self.top_p,
                **kwargs,
            }

            # Add temperature for non-o4/o3 models
            if self.model_name not in [
                "openai/o4-mini",
                "openai/o3-2025-04-16",
            ]:
                completion_params["temperature"] = self.temperature

            # Add tools if specified
            if self.tools_list_dictionary is not None:
                completion_params.update(
                    {
                        "tools": self.tools_list_dictionary,
                        "tool_choice": self.tool_choice,
                        "parallel_tool_calls": self.parallel_tool_calls,
                    }
                )

            if self.functions is not None:
                completion_params.update(
                    {"functions": self.functions}
                )

            if self.base_url is not None:
                completion_params["base_url"] = self.base_url

            # Add modalities if needed
            if self.modalities and len(self.modalities) >= 2:
                completion_params["modalities"] = self.modalities

            # Make the completion call
            response = completion(**completion_params)

            # Handle streaming response
            if self.stream:
                return response  # Return the streaming generator directly

            # Handle tool-based response
            elif self.tools_list_dictionary is not None:
                return self.output_for_tools(response)
            elif self.return_all is True:
                return response.model_dump()
            else:
                # Return standard response content
                return response.choices[0].message.content

        except LiteLLMException as error:
            logger.error(
                f"Error in LiteLLM run: {str(error)} Traceback: {traceback.format_exc()}"
            )
            if "rate_limit" in str(error).lower():
                logger.warning(
                    "Rate limit hit, retrying with exponential backoff..."
                )
                import time

                time.sleep(2)
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
