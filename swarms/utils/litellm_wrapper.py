"""
LiteLLM Wrapper Module

This module provides a comprehensive wrapper around the LiteLLM library for interacting
with various Large Language Models (LLMs) through a unified interface. It supports:

- Multiple model providers (OpenAI, Anthropic, Google, etc.)
- Vision capabilities (image processing)
- Audio processing
- Tool/function calling
- Reasoning models
- Streaming responses
- Batch processing
- Error handling and network connectivity checks

The main class `LiteLLM` provides a simple interface for running LLM tasks with support
for various input modalities and output formats.
"""

import asyncio
import base64
import socket
import traceback
from pathlib import Path
from typing import List, Optional

import litellm
import requests
from litellm import completion, supports_vision
from loguru import logger
from pydantic import BaseModel

from swarms.utils.image_file_b64 import (
    get_image_base64,
    is_base64_encoded,
    save_base64_as_image,
)


class LiteLLMException(Exception):
    """
    Custom exception raised for LiteLLM-specific errors.

    This exception is used to handle errors that occur during LLM operations,
    such as API failures, invalid responses, or configuration issues.
    """


class NetworkConnectionError(Exception):
    """
    Exception raised when network connectivity issues are detected.

    This exception is raised when the wrapper cannot establish a connection
    to the LLM API, either due to network problems, local model server issues,
    or connectivity failures. It provides detailed troubleshooting information
    to help resolve the issue.
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
    A comprehensive wrapper for LiteLLM that provides a unified interface for interacting
    with various Large Language Models (LLMs).

    This class supports multiple model providers including OpenAI, Anthropic, Google,
    and many others through the LiteLLM library. It provides features such as:

    - Text generation with customizable parameters
    - Vision capabilities (image understanding)
    - Audio processing
    - Tool/function calling
    - Reasoning model support
    - Streaming responses
    - Batch processing
    - Automatic error handling and retries

    The class intelligently handles different model requirements, automatically converting
    images to appropriate formats, managing message history, and providing detailed
    error messages for troubleshooting.

    Attributes:
        model_name (str): The name of the model to use.
        system_prompt (str): The system prompt for the conversation.
        stream (bool): Whether to stream responses.
        temperature (float): Sampling temperature for generation.
        max_tokens (int): Maximum number of tokens to generate.
        messages (list): Conversation message history.
        modalities (list): Supported input modalities.

    Example:
        Basic usage:
        ```python
        llm = LiteLLM(model_name="gpt-4", temperature=0.7)
        response = llm.run("What is the capital of France?")
        ```

        With vision:
        ```python
        llm = LiteLLM(model_name="gpt-4-vision-preview")
        response = llm.run("Describe this image", img="path/to/image.jpg")
        ```

        With tools:
        ```python
        tools = [{"type": "function", "function": {...}}]
        llm = LiteLLM(model_name="gpt-4", tools_list_dictionary=tools)
        response = llm.run("Use the weather tool to get today's weather")
        ```
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
            model_name (str, optional): The name of the model to use. Supports any model
                available through LiteLLM (e.g., "gpt-4", "claude-3-opus", "gemini-pro").
                Defaults to "gpt-4.1".
            system_prompt (str, optional): The system prompt to use for the conversation.
                This sets the behavior and context for the model. Defaults to None.
            stream (bool, optional): Whether to stream the output token by token.
                Useful for real-time responses. Defaults to False.
            temperature (float, optional): The sampling temperature for generation.
                Higher values (e.g., 0.9) make output more random, lower values (e.g., 0.1)
                make it more deterministic. Defaults to 0.5.
            max_tokens (int, optional): The maximum number of tokens to generate in the
                response. Defaults to 4000.
            ssl_verify (bool, optional): Whether to verify SSL certificates when making
                API requests. Set to False for self-signed certificates. Defaults to False.
            max_completion_tokens (int, optional): Maximum number of completion tokens.
                Defaults to 4000.
            tools_list_dictionary (List[dict], optional): List of tool/function definitions
                for function calling. Each dict should follow the OpenAI function calling
                format. Defaults to None.
            tool_choice (str, optional): Tool choice strategy. Can be "auto", "none", or
                a specific tool name. Defaults to "auto".
            parallel_tool_calls (bool, optional): Whether to enable parallel tool calls
                when multiple tools are available. Defaults to False.
            audio (str, optional): Path to audio input file. Supported for models with
                audio capabilities. Defaults to None.
            retries (int, optional): Number of retry attempts for failed API calls.
                Defaults to 3.
            verbose (bool, optional): Whether to enable verbose logging for debugging.
                Defaults to False.
            caching (bool, optional): Whether to enable response caching for identical
                requests. Defaults to False.
            mcp_call (bool, optional): Whether this is an MCP (Model Context Protocol) call.
                Affects how tool calls are formatted in the response. Defaults to False.
            top_p (float, optional): Top-p (nucleus) sampling parameter. Controls diversity
                via nucleus sampling. Defaults to 1.0.
            functions (List[dict], optional): Legacy function definitions (deprecated in
                favor of tools_list_dictionary). Defaults to None.
            return_all (bool, optional): Whether to return the complete response object
                instead of just the content. Useful for accessing metadata. Defaults to False.
            base_url (str, optional): Custom base URL for the API endpoint. Useful for
                local models or custom deployments. Defaults to None.
            api_key (str, optional): API key for authentication. If not provided, uses
                environment variables or LiteLLM configuration. Defaults to None.
            api_version (str, optional): API version to use. Some providers support
                multiple API versions. Defaults to None.
            reasoning_effort (str, optional): Reasoning effort level for reasoning-enabled
                models (e.g., "low", "medium", "high"). Defaults to None.
            drop_params (bool, optional): Whether to drop unsupported parameters when
                making API calls. Helps with compatibility across different providers.
                Defaults to True.
            thinking_tokens (int, optional): Budget for thinking tokens in reasoning models.
                Required for Anthropic reasoning models. Defaults to None.
            reasoning_enabled (bool, optional): Whether to enable reasoning mode for
                supported models. Automatically adjusts temperature and other parameters.
                Defaults to False.
            response_format (any, optional): Response format specification (e.g., JSON mode).
                Format depends on the model provider. Defaults to None.
            *args: Additional positional arguments that will be stored and used in run method.
                If a single dictionary is passed, it will be merged into completion parameters.
            **kwargs: Additional keyword arguments that will be stored and used in run method.
                These will be merged into completion parameters with lower priority than
                runtime kwargs passed to the run method.

        Note:
            Parameter priority order (highest to lowest):
            1. Runtime kwargs (passed to run method)
            2. Runtime args (if dictionary, passed to run method)
            3. Init kwargs (passed to __init__)
            4. Init args (if dictionary, passed to __init__)
            5. Default parameters
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
        Check if reasoning is enabled and supported by the model, and adjust parameters accordingly.

        This method validates reasoning configuration and automatically adjusts model parameters
        for optimal reasoning performance. It performs the following checks:

        1. Verifies if the model supports reasoning capabilities
        2. Adjusts temperature to 1.0 for reasoning models (some models require this)
        3. For Anthropic models, ensures thinking_tokens is set and adjusts top_p to 0.95
        4. Logs warnings if reasoning is enabled but not supported by the model

        The method is called automatically when reasoning_enabled is True, but can also be
        called manually to validate configuration.

        Raises:
            No exceptions are raised, but warnings are logged if configuration is invalid.

        Note:
            For Anthropic reasoning models, thinking_tokens is mandatory. If not provided,
            it will be automatically set to max_tokens / 4.
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

        This method merges additional arguments from initialization and runtime into the
        completion parameters dictionary. It handles both dictionary arguments (which are
        merged directly) and other argument types (which are stored for debugging).

        Args:
            completion_params (dict): The completion parameters dictionary to update.
                This dictionary will be modified in-place with merged parameters.
            runtime_args (tuple): Runtime positional arguments passed to the run method.
                If a single dictionary is provided, it will be merged with highest priority.

        Note:
            Priority order for merging:
            1. Runtime args (if dictionary) - highest priority
            2. Init args (if dictionary) - lower priority
            3. Other argument types are stored for debugging purposes
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
        """
        Process and extract tool call information from the LLM response.

        This method handles the output for tool-based responses, supporting both
        MCP (Model Context Protocol) and standard tool call formats. It extracts
        the relevant function name and arguments from the response, handling both
        BaseModel and dictionary outputs.

        Args:
            response (any): The response object returned by the LLM API call.
                Expected to have `response.choices[0].message.tool_calls` containing
                the tool call information.

        Returns:
            dict or list: The format depends on the configuration:
                - If MCP call is enabled and there's a single tool call: Returns a dict
                  with "function" key containing "name" and "arguments"
                - If MCP call is enabled and there are multiple tool calls: Returns a list
                  of dicts, each with "function" key
                - If standard tool calls: Returns the tool_calls directly (as dict or list)
                  after converting BaseModel objects to dictionaries if needed

        Note:
            MCP (Model Context Protocol) format provides a standardized structure for
            tool calls, while standard format uses the provider's native structure.
            The method automatically handles both formats based on the `mcp_call` setting.
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

        This method processes responses from reasoning-enabled models (e.g., o1, o3, Claude
        with thinking) and formats the output to include all reasoning components:
        - Reasoning content (if available)
        - Thinking blocks (for Anthropic models)
        - Tool calls (if any)
        - Main content

        Args:
            response (any): The response object from the LLM API call. Expected to have
                a structure with `response.choices[0].message` containing the message data.

        Returns:
            str: A formatted string containing all reasoning components, thinking blocks,
                tool calls, and the main content, separated by clear sections.

        Note:
            The method checks for various optional attributes in the response:
            - `reasoning_content`: High-level reasoning explanation
            - `thinking_blocks`: Detailed thinking steps (Anthropic models)
            - `tool_calls`: Function/tool calls made during reasoning
            - `content`: The final output content

            All available components are included in the formatted output.
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
        Prepare the messages list for the LLM API call.

        This method constructs the message list that will be sent to the LLM API. It
        handles both text-only and vision (image + text) inputs, ensuring proper
        message formatting for the model.

        Args:
            task (Optional[str]): The text task/prompt. If None, no user message is added.
                Defaults to None.
            img (Optional[str]): Image input (file path, URL, data URI, or base64 string).
                If provided, the task and image are combined into a vision message.
                Defaults to None.

        Returns:
            list: A list of messages formatted for the LLM API. Includes the system
                prompt (if set) and the user message with optional image content.

        Note:
            - If an image is provided, both task and image are included in a single
              vision message via `vision_processing`.
            - If only a task is provided, a simple text message is added.
            - The method creates a copy of the existing messages to avoid modifying
              the original message history.
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

        This method handles Anthropic's specific image format requirements for vision-capable
        models like Claude. It supports multiple image input formats and intelligently chooses
        between direct URL passing and base64 conversion based on the image source and model
        capabilities.

        Args:
            task (str): The text task/prompt associated with the image.
            image (str): The image source. Can be:
                - A file path to a local image file
                - An HTTP/HTTPS URL to a remote image
                - A data URI (data:image/...;base64,...)
                - A raw base64-encoded string
            messages (list): The current message list to append the vision message to.

        Returns:
            list: The updated messages list with the vision message appended.

        Note:
            Anthropic models support specific image formats (JPEG, PNG, GIF, WebP).
            Unsupported formats will be converted to JPEG. The method automatically
            extracts MIME types from data URIs or determines them from file extensions.
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
            # Convert to base64 data URI format (handles file paths, URLs, data URIs, and raw base64)
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

        This method handles OpenAI's specific image format requirements for vision-capable
        models like GPT-4 Vision. It supports multiple image input formats and intelligently
        chooses between direct URL passing and base64 conversion based on the image source
        and model capabilities.

        Args:
            task (str): The text task/prompt associated with the image.
            image (str): The image source. Can be:
                - A file path to a local image file
                - An HTTP/HTTPS URL to a remote image
                - A data URI (data:image/...;base64,...)
                - A raw base64-encoded string
            messages (list): The current message list to append the vision message to.

        Returns:
            list: The updated messages list with the vision message appended.

        Note:
            OpenAI models support a wide range of image formats. The method automatically
            extracts MIME types from data URIs or determines them from file extensions.
            If the model supports direct URLs and the image is a URL, it will be passed
            directly without base64 conversion for better performance.
        """
        # Check if we can use direct URL
        if self._should_use_direct_url(image):
            # Use direct URL without base64 conversion
            vision_message = {
                "type": "image_url",
                "image_url": {"url": image},
            }
        else:
            # Convert to base64 data URI format (handles file paths, URLs, data URIs, and raw base64)
            image_url = get_image_base64(image)

            # Prepare vision message with base64
            vision_message = {
                "type": "image_url",
                "image_url": {"url": image_url},
            }

            # Extract MIME type from data URI or determine from file extension
            mime_type = "image/jpeg"  # default
            if "data:" in image_url and ";base64," in image_url:
                # Extract from data URI
                mime_type = image_url.split(";base64,")[0].split(
                    "data:"
                )[1]
            else:
                # Try to determine from file extension (if it's a file path)
                try:
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
                    mime_type = mime_type_mapping.get(
                        extension, "image/jpeg"
                    )
                except Exception:
                    # If we can't determine, use default
                    pass

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

        This method intelligently decides whether to pass an image URL directly to the API
        or convert it to base64 first. Direct URL passing is more efficient but not all
        models and configurations support it.

        Args:
            image (str): The image source (URL, file path, or base64 string).

        Returns:
            bool: True if we should use direct URL passing, False if we need base64 conversion.

        Note:
            Direct URLs are only used when:
            - The image is an HTTP/HTTPS URL (not a file path or base64)
            - The model is not a local model (Ollama, LlamaCPP, etc.)
            - The model supports vision capabilities
            - The model supports direct URL passing (checked via LiteLLM)
        """
        # Don't use direct URL for base64 strings (data URI or raw base64)
        if is_base64_encoded(image):
            return False

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
        Process the image for the given task with intelligent format handling.

        This method processes vision inputs for LLM models, automatically handling different
        image formats and model-specific requirements. It intelligently chooses between
        direct URL passing and base64 conversion based on the image source and model capabilities.

        The method supports multiple image input formats:
        - File paths: Local image files (e.g., "/path/to/image.jpg")
        - HTTP/HTTPS URLs: Remote image URLs (e.g., "https://example.com/image.png")
        - Data URIs: Base64-encoded images with MIME type (e.g., "data:image/jpeg;base64,...")
        - Raw base64 strings: Base64-encoded image data without data URI prefix

        Args:
            task (str): The text task/prompt associated with the image.
            image (str): The image source in any supported format.
            messages (Optional[list]): The current message list. If None, an empty list is used.

        Returns:
            list: The updated messages list with the vision message appended.

        Note:
            The method automatically routes to model-specific processing:
            - Anthropic models (Claude) use `anthropic_vision_processing`
            - Other models (OpenAI, etc.) use `openai_vision_processing`

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
        Process audio input for the given task.

        This method processes audio files for models that support audio input (e.g., Whisper-based
        models). The audio file is converted to base64 format and added to the message history
        along with the associated text task.

        Args:
            task (str): The text task/prompt associated with the audio input.
            audio (str): The path to the audio file or URL. Supported formats depend on the model,
                but typically include WAV, MP3, and other common audio formats.

        Note:
            The audio is automatically converted to base64 format and added to the message
            history. The format is set to "wav" by default. Ensure your model supports
            audio input before using this method.

        Raises:
            requests.HTTPError: If fetching audio from a URL fails.
            FileNotFoundError: If the local audio file does not exist.
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

        This method uses LiteLLM's built-in `supports_vision` function to verify
        that the model can handle image inputs before processing. It's called
        automatically when an image is provided to prevent errors from unsupported
        models.

        Args:
            img (str, optional): Image path/URL to validate against model capabilities.
                If None, the check is skipped. Defaults to None.

        Raises:
            ValueError: If the model doesn't support vision and an image is provided.
                The error message includes the model name for clarity.

        Note:
            This method only performs the check if `img` is not None. It uses
            LiteLLM's model capability detection to determine vision support.
            Models that support vision include GPT-4 Vision, Claude 3, Gemini Pro Vision,
            and other vision-capable models.
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
        Check if the model name indicates an Anthropic model.

        This method checks if the model name contains "anthropic" (case-insensitive),
        which typically indicates it's a Claude model from Anthropic.

        Args:
            model_name (str): The name of the model to check.

        Returns:
            bool: True if the model appears to be an Anthropic model, False otherwise.

        Example:
            >>> LiteLLM.check_if_model_name_uses_anthropic("claude-3-opus")
            True
            >>> LiteLLM.check_if_model_name_uses_anthropic("gpt-4")
            False
        """
        if "anthropic" in model_name.lower():
            return True
        else:
            return False

    @staticmethod
    def check_if_model_name_uses_openai(model_name: str):
        """
        Check if the model name indicates an OpenAI model.

        This method checks if the model name contains "openai" (case-insensitive),
        which typically indicates it's a model from OpenAI.

        Args:
            model_name (str): The name of the model to check.

        Returns:
            bool: True if the model appears to be an OpenAI model, False otherwise.

        Example:
            >>> LiteLLM.check_if_model_name_uses_openai("gpt-4")
            True
            >>> LiteLLM.check_if_model_name_uses_openai("claude-3-opus")
            False
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
        Run the LLM model for the given task with optional multimodal inputs.

        This is the main method for executing LLM tasks. It supports text-only tasks as well
        as multimodal inputs (images and audio). The method handles message preparation,
        parameter merging, API calls, and response processing.

        Args:
            task (str): The text task or prompt to send to the model. This is the main
                instruction or question for the LLM.
            audio (Optional[str]): Path to an audio file or URL for audio input.
                Supported for models with audio capabilities. Defaults to None.
            img (Optional[str]): Path to an image file, image URL, data URI, or base64
                string for vision input. Supported for vision-capable models. Defaults to None.
            *args: Additional positional arguments. If a single dictionary is passed,
                it will be merged into completion parameters with highest priority.
            **kwargs: Additional keyword arguments that will be merged into completion
                parameters with highest priority (overrides init kwargs). Useful for
                runtime parameter overrides.

        Returns:
            The return type depends on the configuration:
            - str: Text content for standard text responses
            - Generator: Streaming response generator if stream=True
            - dict or list: Tool calls if tools are enabled
            - str: Formatted reasoning output if reasoning is enabled
            - dict: Full response object if return_all=True
            - str: Image file path for Gemini image generation

        Raises:
            NetworkConnectionError: If there are network connectivity issues or the
                API endpoint is unreachable. Includes detailed troubleshooting information.
            LiteLLMException: If there are LiteLLM-specific errors during processing.
            ValueError: If the model doesn't support vision and an image is provided.
            Exception: For other unexpected errors during processing.

        Note:
            Parameter priority order (highest to lowest):
            1. Runtime kwargs (passed to run method)
            2. Runtime args (if dictionary, passed to run method)
            3. Init kwargs (passed to __init__)
            4. Init args (if dictionary, passed to __init__)
            5. Default parameters

        Example:
            Basic text generation:
            ```python
            llm = LiteLLM(model_name="gpt-4")
            response = llm.run("Explain quantum computing")
            ```

            With image:
            ```python
            response = llm.run("Describe this image", img="photo.jpg")
            ```

            With runtime parameter override:
            ```python
            response = llm.run("Write a story", temperature=0.9, max_tokens=2000)
            ```
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
        Make the LiteLLM instance callable, allowing direct invocation.

        This method enables the instance to be called like a function, providing a
        convenient interface for running tasks. It delegates to the `run` method.

        Args:
            task (str): The task to run the model for.
            *args: Additional positional arguments to pass to the run method.
            **kwargs: Additional keyword arguments to pass to the run method.

        Returns:
            The return type depends on the configuration (see `run` method documentation).

        Example:
            ```python
            llm = LiteLLM(model_name="gpt-4")
            response = llm("What is AI?")  # Equivalent to llm.run("What is AI?")
            ```
        """
        return self.run(task, *args, **kwargs)

    def batched_run(self, tasks: List[str], batch_size: int = 10):
        """
        Run multiple tasks in batches synchronously.

        This method processes multiple tasks efficiently by batching them together.
        Tasks are divided into batches of the specified size and processed sequentially.
        This is useful for processing large numbers of tasks while managing API rate
        limits and resource usage.

        Args:
            tasks (List[str]): List of text tasks/prompts to process. Each task will
                be sent to the model independently.
            batch_size (int): The number of tasks to process in each batch. Defaults to 10.
                Adjust based on your API rate limits and processing requirements.

        Returns:
            List[str]: List of responses corresponding to each input task. The order
                of responses matches the order of input tasks.

        Note:
            This method uses asyncio internally for batch processing. The `_process_batch`
            method must be implemented for this to work. Currently, this method references
            `_process_batch` which may need to be implemented separately.

        Example:
            ```python
            llm = LiteLLM(model_name="gpt-4")
            tasks = ["Task 1", "Task 2", "Task 3", "Task 4", "Task 5"]
            responses = llm.batched_run(tasks, batch_size=2)
            # Processes in batches: [Task 1, Task 2], [Task 3, Task 4], [Task 5]
            ```
        """
        logger.info(
            f"Running {len(tasks)} tasks in batches of {batch_size}"
        )
        return asyncio.run(self._process_batch(tasks, batch_size))
