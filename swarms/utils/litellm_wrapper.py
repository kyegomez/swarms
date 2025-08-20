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
        api_version: str = None,
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
            ssl_verify (bool, optional): Whether to verify SSL certificates. Defaults to False.
            max_completion_tokens (int, optional): Maximum completion tokens. Defaults to 4000.
            tools_list_dictionary (List[dict], optional): List of tool definitions. Defaults to None.
            tool_choice (str, optional): Tool choice strategy. Defaults to "auto".
            parallel_tool_calls (bool, optional): Whether to enable parallel tool calls. Defaults to False.
            audio (str, optional): Audio input path. Defaults to None.
            retries (int, optional): Number of retries. Defaults to 0.
            verbose (bool, optional): Whether to enable verbose logging. Defaults to False.
            caching (bool, optional): Whether to enable caching. Defaults to False.
            mcp_call (bool, optional): Whether this is an MCP call. Defaults to False.
            top_p (float, optional): Top-p sampling parameter. Defaults to 1.0.
            functions (List[dict], optional): Function definitions. Defaults to None.
            return_all (bool, optional): Whether to return all response data. Defaults to False.
            base_url (str, optional): Base URL for the API. Defaults to None.
            api_key (str, optional): API key. Defaults to None.
            api_version (str, optional): API version. Defaults to None.
            *args: Additional positional arguments that will be stored and used in run method.
                  If a single dictionary is passed, it will be merged into completion parameters.
            **kwargs: Additional keyword arguments that will be stored and used in run method.
                     These will be merged into completion parameters with lower priority than
                     runtime kwargs passed to the run method.
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
        self.api_version = api_version
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

        # Add system prompt if present
        if self.system_prompt is not None:
            self.messages.append(
                {"role": "system", "content": self.system_prompt}
            )

        # Store additional args and kwargs for use in run method
        self.init_args = args
        self.init_kwargs = kwargs

    def _process_additional_args(
        self, completion_params: dict, runtime_args: tuple
    ):
        """
        Process additional arguments from both initialization and runtime.

        Args:
            completion_params (dict): The completion parameters dictionary to update
            runtime_args (tuple): Runtime positional arguments
        """
        # Process initialization args
        if self.init_args:
            if len(self.init_args) == 1 and isinstance(
                self.init_args[0], dict
            ):
                # If init_args contains a single dictionary, merge it
                completion_params.update(self.init_args[0])
            else:
                # Store other types of init_args for debugging
                completion_params["init_args"] = self.init_args

        # Process runtime args
        if runtime_args:
            if len(runtime_args) == 1 and isinstance(
                runtime_args[0], dict
            ):
                # If runtime_args contains a single dictionary, merge it (highest priority)
                completion_params.update(runtime_args[0])
            else:
                # Store other types of runtime_args for debugging
                completion_params["runtime_args"] = runtime_args

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
        task: Optional[str] = None,
        img: Optional[str] = None,
    ):
        """
        Prepare the messages for the given task.

        Args:
            task (str): The task to prepare messages for.

        Returns:
            list: A list of messages prepared for the task.
        """
        self.check_if_model_supports_vision(img=img)

        # Handle vision case
        if img is not None:
            self.vision_processing(task=task, image=img)

        if task is not None:
            self.messages.append({"role": "user", "content": task})

        return self.messages

    def anthropic_vision_processing(
        self, task: str, image: str, messages: list
    ) -> list:
        """
        Process vision input specifically for Anthropic models.
        Handles Anthropic's specific image format requirements.
        """
        # Check if we can use direct URL
        if self._should_use_direct_url(image):
            # Use direct URL without base64 conversion
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": task},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image,
                            },
                        },
                    ],
                }
            )
        else:
            # Fall back to base64 conversion for local files
            image_url = get_image_base64(image)

            # Extract mime type from the data URI or use default
            mime_type = "image/jpeg"  # default
            if "data:" in image_url and ";base64," in image_url:
                mime_type = image_url.split(";base64,")[0].split(
                    "data:"
                )[1]

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

            # Construct Anthropic vision message with base64
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
        # Check if we can use direct URL
        if self._should_use_direct_url(image):
            # Use direct URL without base64 conversion
            vision_message = {
                "type": "image_url",
                "image_url": {"url": image},
            }
        else:
            # Fall back to base64 conversion for local files
            image_url = get_image_base64(image)

            # Prepare vision message with base64
            vision_message = {
                "type": "image_url",
                "image_url": {"url": image_url},
            }

            # Add format for specific models
            extension = Path(image).suffix.lower()
            mime_type = (
                f"image/{extension[1:]}"
                if extension
                else "image/jpeg"
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

    def _should_use_direct_url(self, image: str) -> bool:
        """
        Determine if we should use direct URL passing instead of base64 conversion.

        Args:
            image (str): The image source (URL or file path)

        Returns:
            bool: True if we should use direct URL, False if we need base64 conversion
        """
        # Only use direct URL for HTTP/HTTPS URLs
        if not image.startswith(("http://", "https://")):
            return False

        # Check for local/custom models that might not support direct URLs
        model_lower = self.model_name.lower()
        local_indicators = [
            "localhost",
            "127.0.0.1",
            "local",
            "custom",
            "ollama",
            "llama-cpp",
        ]

        is_local = any(
            indicator in model_lower for indicator in local_indicators
        ) or (
            self.base_url is not None
            and any(
                indicator in self.base_url.lower()
                for indicator in local_indicators
            )
        )

        if is_local:
            return False

        # Use LiteLLM's supports_vision to check if model supports vision and direct URLs
        try:
            return supports_vision(model=self.model_name)
        except Exception:
            return False

    def vision_processing(
        self, task: str, image: str, messages: Optional[list] = None
    ):
        """
        Process the image for the given task.
        Handles different image formats and model requirements.

        This method now intelligently chooses between:
        1. Direct URL passing (when model supports it and image is a URL)
        2. Base64 conversion (for local files or unsupported models)

        This approach reduces server load and improves performance by avoiding
        unnecessary image downloads and base64 conversions when possible.
        """
        logger.info(f"Processing image for model: {self.model_name}")

        # Log whether we're using direct URL or base64 conversion
        if self._should_use_direct_url(image):
            logger.info(
                f"Using direct URL passing for image: {image[:100]}..."
            )
        else:
            if image.startswith(("http://", "https://")):
                logger.info(
                    "Converting URL image to base64 (model doesn't support direct URLs)"
                )
            else:
                logger.info("Converting local file to base64")

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
        Check if the model supports vision capabilities.

        This method uses LiteLLM's built-in supports_vision function to verify
        that the model can handle image inputs before processing.

        Args:
            img (str, optional): Image path/URL to validate against model capabilities

        Raises:
            ValueError: If the model doesn't support vision and an image is provided
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
            *args: Additional positional arguments. If a single dictionary is passed,
                  it will be merged into completion parameters with highest priority.
            **kwargs: Additional keyword arguments that will be merged into completion
                     parameters with highest priority (overrides init kwargs).

        Returns:
            str: The content of the response from the model.

        Raises:
            Exception: If there is an error in processing the request.

        Note:
            Parameter priority order (highest to lowest):
            1. Runtime kwargs (passed to run method)
            2. Runtime args (if dictionary, passed to run method)
            3. Init kwargs (passed to __init__)
            4. Init args (if dictionary, passed to __init__)
            5. Default parameters
        """
        try:

            self.messages.append({"role": "user", "content": task})

            if img is not None:
                self.messages = self.vision_processing(
                    task=task, image=img
                )

            # Base completion parameters
            completion_params = {
                "model": self.model_name,
                "messages": self.messages,
                "stream": self.stream,
                "max_tokens": self.max_tokens,
                "caching": self.caching,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }

            # Merge initialization kwargs first (lower priority)
            if self.init_kwargs:
                completion_params.update(self.init_kwargs)

            # Merge runtime kwargs (higher priority - overrides init kwargs)
            if kwargs:
                completion_params.update(kwargs)

            if self.api_version is not None:
                completion_params["api_version"] = self.api_version

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

            # Process additional args if any
            self._process_additional_args(completion_params, args)

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
            }

            # Merge initialization kwargs first (lower priority)
            if self.init_kwargs:
                completion_params.update(self.init_kwargs)

            # Merge runtime kwargs (higher priority - overrides init kwargs)
            if kwargs:
                completion_params.update(kwargs)

            # Handle tool-based completion
            if self.tools_list_dictionary is not None:
                completion_params.update(
                    {
                        "tools": self.tools_list_dictionary,
                        "tool_choice": self.tool_choice,
                        "parallel_tool_calls": self.parallel_tool_calls,
                    }
                )

            # Process additional args if any
            self._process_additional_args(completion_params, args)

            # Make the completion call
            response = await acompletion(**completion_params)

            # Handle tool-based response
            if self.tools_list_dictionary is not None:
                return (
                    response.choices[0]
                    .message.tool_calls[0]
                    .function.arguments
                )

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
