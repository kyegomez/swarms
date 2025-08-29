import traceback
from typing import Optional, Callable
import asyncio
import base64
import traceback
import uuid
from pathlib import Path
from typing import List, Optional

import litellm
import requests
from litellm import acompletion, completion, supports_vision
from loguru import logger
from pydantic import BaseModel


class LiteLLMException(Exception):
    """
    Exception for LiteLLM.
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

    def _collect_streaming_chunks(self, streaming_response, callback=None):
        """Helper method to collect chunks from streaming response."""
        chunks = []
        for chunk in streaming_response:
            if hasattr(chunk, "choices") and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                chunks.append(content)
                if callback:
                    callback(content)
        return "".join(chunks)

    def _handle_streaming_response(
        self,
        streaming_response,
        title: str = "LLM Response",
        style: Optional[str] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
        print_on: bool = True,
        verbose: bool = False,
    ) -> str:
        """
        Centralized streaming response handler for all streaming scenarios.
        
        Args:
            streaming_response: The streaming response object
            title: Title for the streaming panel
            style: Style for the panel (optional)
            streaming_callback: Callback for real-time streaming
            print_on: Whether to print the streaming output
            verbose: Whether to enable verbose logging
            
        Returns:
            str: The complete response string
        """
        # Non-streaming response - return as is
        if not (hasattr(streaming_response, "__iter__") and not isinstance(streaming_response, str)):
            return streaming_response
        
        # Handle callback streaming
        if streaming_callback is not None:
            return self._collect_streaming_chunks(streaming_response, streaming_callback)
        
        # Handle silent streaming
        if not print_on:
            return self._collect_streaming_chunks(streaming_response)
        
        # Handle formatted streaming with panel
        from swarms.utils.formatter import formatter
        from loguru import logger
        
        collected_chunks = []
        def on_chunk_received(chunk: str):
            collected_chunks.append(chunk)
            if verbose:
                logger.debug(f"Streaming chunk received: {chunk[:50]}...")

        return formatter.print_streaming_panel(
            streaming_response,
            title=title,
            style=style,
            collect_chunks=True,
            on_chunk_callback=on_chunk_received,
        )

    def run_with_streaming(
        self,
        task: str,
        img: Optional[str] = None,
        audio: Optional[str] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
        title: str = "LLM Response",
        style: Optional[str] = None,
        print_on: bool = True,
        verbose: bool = False,
        *args,
        **kwargs,
    ) -> str:
        """
        Run LLM with centralized streaming handling.
        
        Args:
            task: The task/prompt to send to the LLM
            img: Optional image input
            audio: Optional audio input
            streaming_callback: Callback for real-time streaming
            title: Title for streaming panel
            style: Style for streaming panel
            print_on: Whether to print streaming output
            verbose: Whether to enable verbose logging
            
        Returns:
            str: The complete response
        """
        original_stream = self.stream
        self.stream = True
        
        try:
            # Build kwargs for run method
            run_kwargs = {"task": task, **kwargs}
            if img is not None:
                run_kwargs["img"] = img
            if audio is not None:
                run_kwargs["audio"] = audio
            
            response = self.run(*args, **run_kwargs)
            
            return self._handle_streaming_response(
                response,
                title=title,
                style=style,
                streaming_callback=streaming_callback,
                print_on=print_on,
                verbose=verbose,
            )
        finally:
            self.stream = original_stream

    def run_tool_summary_with_streaming(
        self,
        tool_results: str,
        agent_name: str = "Agent",
        print_on: bool = True,
        verbose: bool = False,
        *args,
        **kwargs,
    ) -> str:
        """
        Run tool summary with streaming support.
        
        Args:
            tool_results: The tool execution results to summarize
            agent_name: Name of the agent for the panel title
            print_on: Whether to print streaming output
            verbose: Whether to enable verbose logging
            
        Returns:
            str: The complete summary response
        """
        return self.run_with_streaming(
            task=f"Please analyze and summarize the following tool execution output:\n\n{tool_results}",
            title=f"Agent: {agent_name} - Tool Summary",
            style="green",
            print_on=print_on,
            verbose=verbose,
            *args,
            **kwargs,
        )

    def handle_string_streaming(
        self,
        response: str,
        print_on: bool = True,
        delay: float = 0.01,
    ) -> None:
        """
        Handle streaming for string responses by simulating streaming output.
        
        Args:
            response: The string response to stream
            print_on: Whether to print the streaming output
            delay: Delay between characters for streaming effect
        """
        if not (print_on and response):
            return
            
        import time
        for char in response:
            print(char, end="", flush=True)
            if delay > 0:
                time.sleep(delay)
        print()  # Newline at the end

    def _process_anthropic_chunk(self, chunk, current_tool_call, tool_call_buffer, tool_calls_in_stream, print_on, verbose):
        """Process Anthropic-style streaming chunks."""
        import json
        from loguru import logger
        
        chunk_type = getattr(chunk, 'type', None)
        full_text_response = ""
        
        if chunk_type == 'content_block_start' and hasattr(chunk, 'content_block') and chunk.content_block.type == 'tool_use':
            tool_name = chunk.content_block.name
            if print_on:
                print(f"\nTool Call: {tool_name}...", flush=True)
            current_tool_call = {"id": chunk.content_block.id, "name": tool_name, "input": ""}
            tool_call_buffer = ""
        
        elif chunk_type == 'content_block_delta' and hasattr(chunk, 'delta'):
            if chunk.delta.type == 'input_json_delta':
                tool_call_buffer += chunk.delta.partial_json
            elif chunk.delta.type == 'text_delta':
                text_chunk = chunk.delta.text
                full_text_response += text_chunk
                if print_on:
                    print(text_chunk, end="", flush=True)

        elif chunk_type == 'content_block_stop' and current_tool_call:
            try:
                tool_input = json.loads(tool_call_buffer)
                current_tool_call["input"] = tool_input
                tool_calls_in_stream.append(current_tool_call)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool arguments: {tool_call_buffer}. Error: {e}")
                # Store the raw buffer for debugging
                current_tool_call["input"] = {"raw_buffer": tool_call_buffer, "error": str(e)}
                tool_calls_in_stream.append(current_tool_call)
            current_tool_call = None
            tool_call_buffer = ""
            
        return full_text_response, current_tool_call, tool_call_buffer
    
    def _process_openai_chunk(self, chunk, tool_calls_in_stream, print_on, verbose):
        """Process OpenAI-style streaming chunks."""
        import json
        full_text_response = ""
        
        if not (hasattr(chunk, 'choices') and chunk.choices):
            return full_text_response
            
        choice = chunk.choices[0]
        if not (hasattr(choice, 'delta') and choice.delta):
            return full_text_response
            
        delta = choice.delta
        
        # Handle text content
        if hasattr(delta, 'content') and delta.content:
            text_chunk = delta.content
            full_text_response += text_chunk
            if print_on:
                print(text_chunk, end="", flush=True)
        
        # Handle tool calls in streaming chunks
        if hasattr(delta, 'tool_calls') and delta.tool_calls:
            for tool_call in delta.tool_calls:
                tool_index = getattr(tool_call, 'index', 0)
                
                # Ensure we have enough slots in the list
                while len(tool_calls_in_stream) <= tool_index:
                    tool_calls_in_stream.append(None)
                
                if hasattr(tool_call, 'function') and tool_call.function:
                    func = tool_call.function
                    
                    # Create new tool call if slot is empty and we have a function name
                    if tool_calls_in_stream[tool_index] is None and hasattr(func, 'name') and func.name:
                        if print_on:
                            print(f"\nTool Call: {func.name}...", flush=True)
                        tool_calls_in_stream[tool_index] = {
                            "id": getattr(tool_call, 'id', f"call_{tool_index}"),
                            "name": func.name,
                            "arguments": ""
                        }
                    
                    # Accumulate arguments
                    if tool_calls_in_stream[tool_index] and hasattr(func, 'arguments') and func.arguments is not None:
                        tool_calls_in_stream[tool_index]["arguments"] += func.arguments
                        
                        if verbose:
                            logger.debug(f"Accumulated arguments for {tool_calls_in_stream[tool_index].get('name', 'unknown')}: '{tool_calls_in_stream[tool_index]['arguments']}'")
                        
                        # Try to parse if we have complete JSON
                        try:
                            args_dict = json.loads(tool_calls_in_stream[tool_index]["arguments"])
                            tool_calls_in_stream[tool_index]["input"] = args_dict
                            tool_calls_in_stream[tool_index]["arguments_complete"] = True
                            if verbose:
                                logger.info(f"Complete tool call for {tool_calls_in_stream[tool_index]['name']} with args: {args_dict}")
                        except json.JSONDecodeError:
                            # Continue accumulating - JSON might be incomplete
                            if verbose:
                                logger.debug(f"Incomplete JSON for {tool_calls_in_stream[tool_index].get('name', 'unknown')}: {tool_calls_in_stream[tool_index]['arguments'][:100]}...")
                            
        return full_text_response

    def parse_streaming_chunks_with_tools(
        self,
        stream,
        agent_name: str = "Agent",
        print_on: bool = True,
        verbose: bool = False,
    ) -> tuple:
        """
        Parse streaming chunks and extract both text and tool calls.
        
        Args:
            stream: The streaming response object
            agent_name: Name of the agent for printing
            print_on: Whether to print streaming output
            verbose: Whether to enable verbose logging
            
        Returns:
            tuple: (full_text_response, tool_calls_list)
        """
        full_text_response = ""
        tool_calls_in_stream = []
        current_tool_call = None
        tool_call_buffer = ""

        if print_on:
            print(f"{agent_name}: ", end="", flush=True)

        # Process streaming chunks in real-time
        try:
            for chunk in stream:
                if verbose:
                    logger.debug(f"Processing streaming chunk: {type(chunk)}")
                
                # Try Anthropic-style processing first
                anthropic_result = self._process_anthropic_chunk(
                    chunk, current_tool_call, tool_call_buffer, tool_calls_in_stream, print_on, verbose
                )
                if anthropic_result[0]:  # If text was processed
                    text_chunk, current_tool_call, tool_call_buffer = anthropic_result
                    full_text_response += text_chunk
                    continue
                
                # If not Anthropic, try OpenAI-style processing
                openai_text = self._process_openai_chunk(chunk, tool_calls_in_stream, print_on, verbose)
                if openai_text:
                    full_text_response += openai_text
        except Exception as e:
            logger.error(f"Error processing streaming chunks: {e}")
            if print_on:
                print(f"\n[Streaming Error: {e}]")
            return full_text_response, tool_calls_in_stream

        if print_on:
            print()  # Newline after streaming text

        return full_text_response, tool_calls_in_stream

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
            elif "gemini" in self.model_name.lower():
                return gemini_output_img_handler(response)
            else:
                # For non-Gemini models, return the content directly
                return response.choices[0].message.content

        except LiteLLMException as error:
            logger.error(
                f"Error in LiteLLM run: {str(error)} Traceback: {traceback.format_exc()}"
            )
            if "rate_limit" in str(error).lower():
                logger.warning(
                    "Rate limit hit, retrying with exponential backoff..."
                )
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
            # Extract image parameter from kwargs if present
            img = kwargs.pop("img", None) if "img" in kwargs else None
            messages = self._prepare_messages(task=task, img=img)

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
            # Standard completion
            response = await acompletion(**completion_params)

            print(response)
            return response
            elif self.return_all is True:
                return response.model_dump()
            elif "gemini" in self.model_name.lower():
                return gemini_output_img_handler(response)
            else:
                # For non-Gemini models, return the content directly
                return response.choices[0].message.content

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
