import asyncio
import base64
import traceback
import uuid
from pathlib import Path
from typing import List, Optional
import socket

import litellm
from pydantic import BaseModel
import requests
from litellm import completion, supports_vision
from loguru import logger


class LiteLLMException(Exception):
    """
    Exception for LiteLLM.
    """


class NetworkConnectionError(Exception):
    """
    Exception raised when network connectivity issues are detected.
    """


def get_audio_base64(audio_source: str) -> str:
    """
    Convert audio data from a URL or local file path to a base64-encoded string.

    This function supports both remote (HTTP/HTTPS) and local audio sources. If the source is a URL,
    it fetches the audio data via HTTP. If the source is a local file path, it reads the file directly.

    Args:
        audio_source (str): The path or URL to the audio file.

    Returns:
        str: The base64-encoded string of the audio data.

    Raises:
        requests.HTTPError: If fetching audio from a URL fails.
        FileNotFoundError: If the local audio file does not exist.
    """
    if audio_source.startswith(("http://", "https://")):
        response = requests.get(audio_source)
        response.raise_for_status()
        audio_data = response.content
    else:
        with open(audio_source, "rb") as file:
            audio_data = file.read()

    encoded_string = base64.b64encode(audio_data).decode("utf-8")
    return encoded_string


def get_image_base64(image_source: str) -> str:
    """
    Convert image data from a URL, local file path, or data URI to a base64-encoded string in data URI format.

    If the input is already a data URI, it is returned unchanged. Otherwise, the image is loaded from the
    specified source, encoded as base64, and returned as a data URI with the appropriate MIME type.

    Args:
        image_source (str): The path, URL, or data URI of the image.

    Returns:
        str: The image as a base64-encoded data URI string.

    Raises:
        requests.HTTPError: If fetching the image from a URL fails.
        FileNotFoundError: If the local image file does not exist.
    """
    if image_source.startswith("data:image"):
        return image_source

    if image_source.startswith(("http://", "https://")):
        response = requests.get(image_source)
        response.raise_for_status()
        image_data = response.content
    else:
        with open(image_source, "rb") as file:
            image_data = file.read()

    extension = Path(image_source).suffix.lower()
    mime_type_mapping = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".svg": "image/svg+xml",
    }
    mime_type = mime_type_mapping.get(extension, "image/jpeg")
    encoded_string = base64.b64encode(image_data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_string}"


def save_base64_as_image(
    base64_data: str,
    output_dir: str = "images",
) -> str:
    """
    Decode base64-encoded image data and save it as an image file in the specified directory.

    This function supports both raw base64 strings and data URIs (data:image/...;base64,...).
    The image format is determined from the MIME type if present, otherwise defaults to JPEG.
    The image is saved with a randomly generated filename.

    Args:
        base64_data (str): The base64-encoded image data, either as a raw string or a data URI.
        output_dir (str, optional): Directory to save the image file. Defaults to "images".
            If None, saves to the current working directory.

    Returns:
        str: The full path to the saved image file.

    Raises:
        ValueError: If the base64 data is not a valid data URI or is otherwise invalid.
        IOError: If the image cannot be written to disk.
    """
    import os

    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    if base64_data.startswith("data:image"):
        try:
            header, encoded_data = base64_data.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]
        except (ValueError, IndexError):
            raise ValueError("Invalid data URI format")
    else:
        encoded_data = base64_data
        mime_type = "image/jpeg"

    mime_to_extension = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
        "image/svg+xml": ".svg",
    }
    extension = mime_to_extension.get(mime_type, ".jpg")
    filename = f"{uuid.uuid4()}{extension}"
    file_path = os.path.join(output_dir, filename)

    try:
        logger.debug(
            f"Attempting to decode base64 data of length: {len(encoded_data)}"
        )
        logger.debug(
            f"Base64 data (first 100 chars): {encoded_data[:100]}..."
        )
        image_data = base64.b64decode(encoded_data)
        with open(file_path, "wb") as f:
            f.write(image_data)
        logger.info(f"Image saved successfully to: {file_path}")
        return file_path
    except Exception as e:
        logger.error(
            f"Base64 decoding failed. Data length: {len(encoded_data)}"
        )
        logger.error(
            f"First 100 chars of data: {encoded_data[:100]}..."
        )
        raise IOError(f"Failed to save image: {str(e)}")


def gemini_output_img_handler(response: any):
    """
    Handle Gemini model output that may contain a base64-encoded image string.

    If the response content is a base64-encoded image (i.e., a string starting with a known image data URI prefix),
    this function saves the image to disk and returns the file path. Otherwise, it returns the content as is.

    Args:
        response (any): The response object from the Gemini model. It is expected to have
            a structure such that `response.choices[0].message.content` contains the output.

    Returns:
        str: The file path to the saved image if the content is a base64 image, or the original content otherwise.
    """
    response_content = response.choices[0].message.content

    base64_prefixes = [
        "data:image/jpeg;base64,",
        "data:image/jpg;base64,",
        "data:image/png;base64,",
        "data:image/gif;base64,",
        "data:image/webp;base64,",
        "data:image/bmp;base64,",
        "data:image/tiff;base64,",
        "data:image/svg+xml;base64,",
    ]

    if isinstance(response_content, str) and any(
        response_content.strip().startswith(prefix)
        for prefix in base64_prefixes
    ):
        return save_base64_as_image(base64_data=response_content)
    else:
        return response_content


class LiteLLM:
    """
    This class represents a LiteLLM.
    It is used to interact with the LLM model for various tasks.
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1",
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
        mcp_call: bool = False,
        top_p: float = 1.0,
        functions: List[dict] = None,
        return_all: bool = False,
        base_url: str = None,
        api_key: str = None,
        api_version: str = None,
        reasoning_effort: str = None,
        drop_params: bool = True,
        thinking_tokens: int = None,
        reasoning_enabled: bool = False,
        response_format: any = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the LiteLLM with the given parameters.

        Args:
            model_name (str, optional): The name of the model to use. Defaults to "gpt-4.1".
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
        self.reasoning_effort = reasoning_effort
        self.thinking_tokens = thinking_tokens
        self.reasoning_enabled = reasoning_enabled
        self.verbose = verbose
        self.response_format = response_format
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

        litellm.drop_params = drop_params

        # Add system prompt if present
        if self.system_prompt is not None:
            self.messages.append(
                {"role": "system", "content": self.system_prompt}
            )

        # Store additional args and kwargs for use in run method
        self.init_args = args
        self.init_kwargs = kwargs

        # if self.reasoning_enabled is True:
        #     self.reasoning_check()

    def reasoning_check(self):
        """
        Check if reasoning is enabled and supported by the model, and adjust temperature accordingly.

        If reasoning is enabled and the model supports reasoning, set temperature to 1 for optimal reasoning.
        Also logs information or warnings based on the model's reasoning support and configuration.
        """
        """
        Check if reasoning is enabled and supported by the model, and adjust temperature, thinking_tokens, and top_p accordingly.

        This single-line version combines all previous checks and actions for reasoning-enabled models, including Anthropic-specific logic.
        """
        if self.reasoning_enabled:
            supports_reasoning = litellm.supports_reasoning(
                model=self.model_name
            )
            uses_anthropic = self.check_if_model_name_uses_anthropic(
                model_name=self.model_name
            )
            if supports_reasoning:
                logger.info(
                    f"Model {self.model_name} supports reasoning and reasoning enabled is set to {self.reasoning_enabled}. Temperature will be set to 1 for better reasoning as some models may not work with low temperature."
                )
                self.temperature = 1
            else:
                logger.warning(
                    f"Model {self.model_name} does not support reasoning and reasoning enabled is set to {self.reasoning_enabled}. Temperature will not be set to 1."
                )
                logger.warning(
                    f"Model {self.model_name} may or may not support reasoning and reasoning enabled is set to {self.reasoning_enabled}"
                )
            if uses_anthropic:
                if self.thinking_tokens is None:
                    logger.info(
                        f"Model {self.model_name} is an Anthropic model and reasoning enabled is set to {self.reasoning_enabled}. Thinking tokens is mandatory for Anthropic models."
                    )
                    self.thinking_tokens = self.max_tokens / 4
                logger.info(
                    "top_p must be greater than 0.95 for Anthropic models with reasoning enabled"
                )
                self.top_p = 0.95

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

    # def output_for_tools(self, response: any):
    #     """
    #     Process tool calls from the LLM response and return formatted output.

    #     Args:
    #         response: The response object from the LLM API call

    #     Returns:
    #         dict or list: Formatted tool call data, or default response if no tool calls
    #     """
    #     try:
    #         # Convert response to dict if it's a Pydantic model
    #         if hasattr(response, "model_dump"):
    #             response_dict = response.model_dump()
    #         else:
    #             response_dict = response

    #         print(f"Response dict: {response_dict}")

    #         # Check if tool_calls exists and is not None
    #         if (
    #             response_dict.get("choices")
    #             and response_dict["choices"][0].get("message")
    #             and response_dict["choices"][0]["message"].get(
    #                 "tool_calls"
    #             )
    #             and len(
    #                 response_dict["choices"][0]["message"][
    #                     "tool_calls"
    #                 ]
    #             )
    #             > 0
    #         ):
    #             tool_call = response_dict["choices"][0]["message"][
    #                 "tool_calls"
    #             ][0]
    #             if "function" in tool_call:
    #                 return {
    #                     "function": {
    #                         "name": tool_call["function"].get(
    #                             "name", ""
    #                         ),
    #                         "arguments": tool_call["function"].get(
    #                             "arguments", "{}"
    #                         ),
    #                     }
    #                 }
    #             else:
    #                 # Handle case where tool_call structure is different
    #                 return tool_call
    #         else:
    #             # Return a default response when no tool calls are present
    #             logger.warning(
    #                 "No tool calls found in response, returning default response"
    #             )
    #             return {
    #                 "function": {
    #                     "name": "no_tool_call",
    #                     "arguments": "{}",
    #                 }
    #             }
    #     except Exception as e:
    #         logger.error(f"Error processing tool calls: {str(e)} Traceback: {traceback.format_exc()}")

    def output_for_tools(self, response: any):
        """
        Process and extract tool call information from the LLM response.

        This function handles the output for tool-based responses, supporting both
        MCP (Multi-Call Protocol) and standard tool call formats. It extracts the
        relevant function name and arguments from the response, handling both
        BaseModel and dictionary outputs.

        Args:
            response (any): The response object returned by the LLM API call.

        Returns:
            dict or list: A dictionary containing the function name and arguments
                if MCP call is used, or the tool calls output (as a dict or list)
                for standard tool call responses.
        """
        if self.mcp_call is True:
            tool_calls = response.choices[0].message.tool_calls

            # Check if there are multiple tool calls
            if len(tool_calls) > 1:
                # Return all tool calls if there are multiple
                return [
                    {
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }
                    }
                    for tool_call in tool_calls
                ]
            else:
                # Single tool call
                out = tool_calls[0].function
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

    def output_for_reasoning(self, response: any):
        """
        Handle output for reasoning models, formatting reasoning content and thinking blocks.

        Args:
            response: The response object from the LLM API call

        Returns:
            str: Formatted string containing reasoning content, thinking blocks, and main content
        """
        output_parts = []

        # Check if reasoning content is available
        if (
            hasattr(response.choices[0].message, "reasoning_content")
            and response.choices[0].message.reasoning_content
        ):
            output_parts.append(
                f"Reasoning Content:\n{response.choices[0].message.reasoning_content}\n"
            )

        # Check if thinking blocks are available (Anthropic models)
        if (
            hasattr(response.choices[0].message, "thinking_blocks")
            and response.choices[0].message.thinking_blocks
        ):
            output_parts.append("Thinking Blocks:")
            for i, block in enumerate(
                response.choices[0].message.thinking_blocks, 1
            ):
                block_type = block.get("type", "")
                thinking = block.get("thinking", "")
                output_parts.append(
                    f"Block {i} (Type: {block_type}):"
                )
                output_parts.append(f"  Thinking: {thinking}")
                output_parts.append("")

        # Include tools if available
        if (
            hasattr(response.choices[0].message, "tool_calls")
            and response.choices[0].message.tool_calls
        ):
            output_parts.append(
                f"Tools:\n{self.output_for_tools(response)}\n"
            )

        # Always include the main content
        content = response.choices[0].message.content
        if content:
            output_parts.append(f"Content:\n{content}")

        # Join all parts into a single string
        return "\n".join(output_parts)

    def _prepare_messages(
        self,
        task: Optional[str] = None,
        img: Optional[str] = None,
    ):
        """
        Prepare the messages for the given task.

        Args:
            task (str): The task to prepare messages for.
            img (str, optional): Image input if any. Defaults to None.

        Returns:
            list: A list of messages prepared for the task.
        """
        # Start with a fresh copy of messages to avoid duplication
        messages = self.messages.copy()

        # Check if model supports vision if image is provided
        if img is not None:
            self.check_if_model_supports_vision(img=img)
            # Handle vision case - this already includes both task and image
            messages = self.vision_processing(
                task=task, image=img, messages=messages
            )
        elif task is not None:
            # Only add task message if no image (since vision_processing handles both)
            messages.append({"role": "user", "content": task})

        return messages

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

            # Map common image extensions to proper MIME types
            mime_type_mapping = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
                ".bmp": "image/bmp",
                ".tiff": "image/tiff",
                ".svg": "image/svg+xml",
            }

            mime_type = mime_type_mapping.get(extension, "image/jpeg")
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
        # Ensure messages is a list
        if messages is None:
            messages = []

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

    @staticmethod
    def check_if_model_name_uses_anthropic(model_name: str):
        """
        Check if the model name uses Anthropic.
        """
        if "anthropic" in model_name.lower():
            return True
        else:
            return False

    @staticmethod
    def check_if_model_name_uses_openai(model_name: str):
        """
        Check if the model name uses OpenAI.
        """
        if "openai" in model_name.lower():
            return True
        else:
            return False

    @staticmethod
    def check_internet_connection(
        host: str = "8.8.8.8", port: int = 53, timeout: int = 3
    ) -> bool:
        """
        Check if there is an active internet connection.

        This method attempts to establish a socket connection to a DNS server
        (default is Google's DNS at 8.8.8.8) to verify internet connectivity.

        Args:
            host (str, optional): The host to connect to for checking connectivity.
                Defaults to "8.8.8.8" (Google DNS).
            port (int, optional): The port to use for the connection. Defaults to 53 (DNS).
            timeout (int, optional): Connection timeout in seconds. Defaults to 3.

        Returns:
            bool: True if internet connection is available, False otherwise.
        """
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(
                (host, port)
            )
            return True
        except (socket.error, socket.timeout):
            return False

    @staticmethod
    def is_local_model(
        model_name: str, base_url: Optional[str] = None
    ) -> bool:
        """
        Determine if the model is a local model (e.g., Ollama, LlamaCPP).

        Args:
            model_name (str): The name of the model to check.
            base_url (str, optional): The base URL if specified. Defaults to None.

        Returns:
            bool: True if the model is a local model, False otherwise.
        """
        local_indicators = [
            "ollama",
            "llama-cpp",
            "local",
            "localhost",
            "127.0.0.1",
            "custom",
        ]

        model_lower = model_name.lower()
        is_local_model = any(
            indicator in model_lower for indicator in local_indicators
        )

        is_local_url = base_url is not None and any(
            indicator in base_url.lower()
            for indicator in local_indicators
        )

        return is_local_model or is_local_url

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
            # Prepare messages properly - this handles both task and image together
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

            if self.response_format is not None:
                completion_params["response_format"] = (
                    self.response_format
                )

            # Add modalities if needed
            if self.modalities and len(self.modalities) >= 2:
                completion_params["modalities"] = self.modalities

            if (
                self.reasoning_effort is not None
                and litellm.supports_reasoning(model=self.model_name)
                is True
            ):
                completion_params["reasoning_effort"] = (
                    self.reasoning_effort
                )

            if (
                self.reasoning_enabled is True
                and self.thinking_tokens is not None
            ):
                thinking = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_tokens,
                }
                completion_params["thinking"] = thinking

            # Process additional args if any
            self._process_additional_args(completion_params, args)

            # Make the completion call
            response = completion(**completion_params)
            # print(response)

            # Validate response
            if not response:
                logger.error(
                    "Received empty response from completion call"
                )
                return None

            # Handle streaming response
            if self.stream:
                return response  # Return the streaming generator directly

            # Handle reasoning model output
            elif (
                self.reasoning_enabled
                and self.reasoning_effort is not None
            ):
                return self.output_for_reasoning(response)

            # Handle tool-based response
            elif self.tools_list_dictionary is not None:
                result = self.output_for_tools(response)
                return result
            elif self.return_all is True:
                return response.model_dump()
            elif "gemini" in self.model_name.lower():
                return gemini_output_img_handler(response)
            else:
                return response.choices[0].message.content

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException,
            ConnectionError,
            TimeoutError,
        ) as network_error:
            # Check if this is a local model
            if self.is_local_model(self.model_name, self.base_url):
                error_msg = (
                    f"Network error connecting to local model '{self.model_name}': {str(network_error)}\n\n"
                    "Troubleshooting steps:\n"
                    "1. Ensure your local model server (e.g., Ollama, LlamaCPP) is running\n"
                    "2. Verify the base_url is correct and accessible\n"
                    "3. Check that the model is properly loaded and available\n"
                )
                logger.error(error_msg)
                raise NetworkConnectionError(
                    error_msg
                ) from network_error

            # Check internet connectivity
            has_internet = self.check_internet_connection()

            if not has_internet:
                error_msg = (
                    f"No internet connection detected while trying to use model '{self.model_name}'.\n\n"
                    "Possible solutions:\n"
                    "1. Check your internet connection and try again\n"
                    "2. Reconnect to your network\n"
                    "3. Use a local model instead (e.g., Ollama):\n"
                    "   - Install Ollama from https://ollama.ai\n"
                    "   - Run: ollama pull llama2\n"
                    "   - Use model_name='ollama/llama2' in your LiteLLM configuration\n"
                    "\nExample:\n"
                    "  model = LiteLLM(model_name='ollama/llama2')\n"
                )
                logger.error(error_msg)
                raise NetworkConnectionError(
                    error_msg
                ) from network_error
            else:
                # Internet is available but request failed
                error_msg = (
                    f"Network error occurred while connecting to '{self.model_name}': {str(network_error)}\n\n"
                    "Possible causes:\n"
                    "1. The API endpoint may be temporarily unavailable\n"
                    "2. Connection timeout or slow network\n"
                    "3. Firewall or proxy blocking the connection\n"
                    "\nConsider using a local model as a fallback:\n"
                    "  model = LiteLLM(model_name='ollama/llama2')\n"
                )
                logger.error(error_msg)
                raise NetworkConnectionError(
                    error_msg
                ) from network_error

        except LiteLLMException as error:
            logger.error(
                f"Error in LiteLLM run: {str(error)} Traceback: {traceback.format_exc()}"
            )
            raise

        except Exception as error:
            logger.error(
                f"Unexpected error in LiteLLM run: {str(error)} Traceback: {traceback.format_exc()}"
            )
            raise

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
