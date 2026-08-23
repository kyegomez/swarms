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

import socket
import traceback
from functools import lru_cache
from typing import List, Optional, Union

import litellm
import requests
from litellm import acompletion, completion, supports_vision
from loguru import logger
from pydantic import BaseModel

from swarms.utils.formatter import formatter
from swarms.utils.image_file_b64 import (
    get_image_base64,
    get_media_base64,
    is_base64_encoded,
    save_base64_as_image,
)


@lru_cache(maxsize=None)
def _model_supports_vision(model: str) -> bool:
    """Cached litellm.supports_vision lookup (pure function of model name)."""
    return supports_vision(model=model)


@lru_cache(maxsize=None)
def _model_supports_reasoning(model: str) -> bool:
    """Cached litellm.supports_reasoning lookup (pure function of model name)."""
    return litellm.supports_reasoning(model=model)


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
        model_name: str = "gpt-5.4",
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
        prompt_caching: bool = False,
        cache_config: dict = None,
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
        agent_name: str = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the LiteLLM with the given parameters.

        Args:
            model_name (str, optional): The name of the model to use. Supports any model
                available through LiteLLM (e.g., "gpt-4", "claude-3-opus", "gemini-pro").
                Defaults to "gpt-5.4".
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
            caching (bool, optional): Whether to enable LiteLLM response caching for
                identical requests (a full-response cache). Defaults to False.
            prompt_caching (bool, optional): Whether to enable provider-side prompt
                caching. When True, ephemeral ``cache_control`` breakpoints are added to
                the system prompt and the final message so the large, stable prefix of
                each request is cached and re-billed at a discount (Anthropic model
                family: Claude on Anthropic / Bedrock / Vertex). Providers that cache
                automatically (e.g. OpenAI) are left untouched. Defaults to False.
            cache_config (dict, optional): Fine-grained prompt-caching options; only
                consulted when ``prompt_caching=True``. Recognized keys (all optional):

                    ttl (str): "5m" (default) or "1h" for Anthropic's extended cache.
                    cache_system_prompt (bool): cache the system prefix (default True).
                    cache_messages (bool): cache through the last message (default True).
                    cache_tools (bool): cache the tool definitions block (default True).
                    override (bool): force cache_control injection on/off regardless of
                        the detected provider — e.g. to opt Gemini/Vertex in, or a
                        custom alias out. Default None (auto-detect: Anthropic only).
                    prompt_cache_key (str): OpenAI routing hint for higher hit rates.
                    prompt_cache_retention (str): OpenAI cache TTL ("in_memory" | "24h").

                Defaults to None (all defaults above apply).
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
        self.prompt_caching = prompt_caching
        self.cache_config = cache_config or {}
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
        self.agent_name = agent_name
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

        # Add system prompt if present (Anthropic rejects empty system blocks)
        if isinstance(self.system_prompt, str):
            self.system_prompt = self.system_prompt.strip()
            if self.system_prompt:
                self.messages.append(
                    {"role": "system", "content": self.system_prompt}
                )
        elif self.system_prompt is not None:
            # Non-string system prompt (rare) - only include if not None
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
            supports_reasoning = _model_supports_reasoning(
                self.model_name
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
        message = response.choices[0].message
        thinking_parts = []

        has_thinking_blocks = bool(
            hasattr(message, "thinking_blocks")
            and message.thinking_blocks
        )

        # Prefer thinking_blocks for Anthropic; fall back to reasoning_content
        # for all other providers. Both fields carry the same text on Anthropic,
        # so only collect one to avoid duplicates.
        if has_thinking_blocks:
            for block in message.thinking_blocks:
                thinking = block.get("thinking", "")
                if thinking:
                    thinking_parts.append(thinking)
        elif (
            hasattr(message, "reasoning_content")
            and message.reasoning_content
        ):
            thinking_parts.append(message.reasoning_content)

        # Display all thinking content in a dedicated panel
        if thinking_parts:
            title = (
                f"{self.agent_name} | Thinking"
                if self.agent_name
                else "Thinking"
            )
            formatter.print_thinking_panel(
                "\n\n".join(thinking_parts),
                title=title,
            )

        # Return only the main content so memory stays clean
        return message.content or ""

    def _prepare_messages(
        self,
        task: Optional[str] = None,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        messages: Optional[List[dict]] = None,
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
            messages (Optional[List[dict]]): A prebuilt conversation body in
                OpenAI chat format - user/assistant turns, ``tool_calls`` on
                assistant messages, and ``{"role": "tool", "tool_call_id": ...}``
                results. When given, it is appended after the system prompt and
                ``task`` is ignored. This is the path an agentic loop uses to
                preserve real tool-call structure instead of flattening the
                conversation into a single user string.

        Returns:
            list: A list of messages formatted for the LLM API. Includes the system
                prompt (if set) and the user message with optional image content.

        Note:
            - Some higher-level orchestrators may wrap prompts in a pseudo-format like:

                  "System: ...\\n\\nHuman: ..."

              For Anthropic-compatible backends this can be auto-parsed into separate
              system/user blocks. However, if the "System:" section is effectively
              empty (e.g. "System: \\n\\nHuman: Say hi"), Anthropic will reject the
              request with "system: text content blocks must be non-empty".
              To avoid this, we normalize such prompts by stripping the empty
              system section and keeping only the human text.
            - If an image is provided, both task and image are included in a single
              vision message via `vision_processing`.
            - If only a task is provided, a simple text message is added.
            - The method creates a copy of the existing messages to avoid modifying
              the original message history.
        """
        # Normalize orchestrator-style "System:/Human:" prompts where the System
        # section is effectively empty (e.g. "System: \\n\\nHuman: Say hi").
        # This prevents Anthropic API errors about empty system text blocks.
        if isinstance(task, str) and "Human:" in task:
            stripped = task.lstrip()
            if stripped.startswith("System:"):
                try:
                    system_part, human_part = stripped.split(
                        "Human:", 1
                    )
                    # Remove the leading "System:" label and surrounding whitespace
                    system_content = system_part[
                        len("System:") :
                    ].strip()
                    if not system_content:
                        # No real system content → keep only the human text
                        task = human_part.strip()
                except ValueError:
                    # If splitting fails for any reason, fall back to original task
                    pass

        # Start with a fresh copy of this instance's own turns (the system
        # prompt) to avoid duplication. Also drop any empty system blocks to
        # satisfy Anthropic validation. Named `base` rather than `messages` so
        # it cannot shadow the `messages` parameter below.
        base = []
        for m in self.messages:
            if not isinstance(m, dict):
                continue
            if m.get("role") != "system":
                base.append(m)
                continue
            content = m.get("content")
            if content is None:
                continue
            if isinstance(content, str) and not content.strip():
                continue
            base.append(m)

        # A prebuilt conversation body replaces the single-user-turn path
        # entirely: the caller has already structured the turns, so the only
        # thing to add is this instance's system prompt.
        if messages is not None:
            prepared = base + list(messages)
            if self.prompt_caching and prepared:
                self._apply_prompt_caching(prepared)
            return prepared

        messages = base

        # Check if model supports vision if any image is provided
        images = [
            i for i in ([img] if img else []) + (imgs or []) if i
        ]
        if images:
            self.check_if_model_supports_vision(img=images[0])
            # Handle vision case - this includes the task and every image
            messages = self.vision_processing(
                task=task, image=images, messages=messages
            )
        elif task is not None:
            # Only add task message if no image (since vision_processing handles both)
            messages.append({"role": "user", "content": task})

        # Insert prompt-caching breakpoints on the stable prefix if enabled.
        if self.prompt_caching and messages:
            self._apply_prompt_caching(messages)

        return messages

    def _cache_opt(self, key: str, default):
        """Read a single option out of ``self.cache_config`` with a default."""
        if isinstance(self.cache_config, dict):
            val = self.cache_config.get(key, default)
            return default if val is None else val
        return default

    def _supports_prompt_caching(self) -> bool:
        """
        Whether to inject ``cache_control`` breakpoints for the current model.

        ``cache_control`` ephemeral blocks are the explicit prompt-caching
        mechanism for the **Anthropic model family** — Claude on the Anthropic
        API, on AWS Bedrock, and on Google Vertex AI. Only those are marked by
        default.

        Every other provider is intentionally left alone:
          * OpenAI / xAI / Deepseek cache automatically (no markers needed).
          * Gemini / Google AI Studio uses its own context-caching API, not
            ``cache_control`` — injecting the blocks there corrupts the request
            ("contents is not specified"), so we must not touch it.

        Override: ``cache_config={"override": True/False}`` forces injection on
        or off regardless of the model — useful for proxies, custom aliases, or
        to opt Gemini/Vertex into the ``cache_control`` path (which LiteLLM's
        docs say it supports, though behavior varies by version).

        Note: ``litellm.utils.supports_prompt_caching`` is deliberately NOT used
        here — it returns True for providers (e.g. Gemini) whose caching is not
        driven by ``cache_control``, which would break those requests.
        """
        override = self._cache_opt("override", None)
        if override is not None:
            return bool(override)
        return self._is_anthropic_model()

    def _is_anthropic_model(self) -> bool:
        """
        Whether the configured model is in the Anthropic Claude family, incl.
        "bedrock/anthropic.claude-*" and "vertex_ai/claude-*". Excludes
        Gemini/Vertex-Gemini and OpenAI.
        """
        name = (self.model_name or "").lower()
        return "claude" in name or "anthropic" in name

    def _cache_control_value(self) -> dict:
        """
        Build the ``cache_control`` marker, honoring the configured TTL.

        Default TTL is 5 minutes (``{"type": "ephemeral"}``); ``ttl="1h"`` opts
        into Anthropic's 1-hour cache (2x write cost, survives longer gaps).
        """
        marker = {"type": "ephemeral"}
        ttl = self._cache_opt("ttl", "5m")
        if ttl and ttl != "5m":
            marker["ttl"] = ttl
        return marker

    def _add_cache_control(self, message: dict) -> None:
        """
        Attach a ``cache_control`` breakpoint to a single message in-place,
        converting plain string content into the block form that
        LiteLLM/Anthropic require for cache markers.
        """
        content = message.get("content")
        if content is None:
            return

        marker = self._cache_control_value()
        if isinstance(content, str):
            if not content.strip():
                return
            message["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": marker,
                }
            ]
        elif isinstance(content, list) and content:
            # Prefer marking the last text block; otherwise the last block.
            for block in reversed(content):
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                ):
                    block["cache_control"] = marker
                    return
            last = content[-1]
            if isinstance(last, dict):
                last["cache_control"] = marker

    def _apply_prompt_caching(self, messages: list) -> None:
        """
        Insert ``cache_control`` breakpoints for provider-side prompt caching.

        By default two breakpoints are used (Anthropic allows up to four): the
        system prompt (stable across the whole run) and the final message (so the
        growing conversation prefix is cached incrementally across loops). Each is
        individually toggleable via ``cache_config``:

            cache_config = {
                "cache_system_prompt": True,   # cache the system prefix
                "cache_messages": True,        # cache through the last message
            }

        No-op for providers that do not support ``cache_control``; those (e.g.
        OpenAI) cache automatically and need no markers.
        """
        if not self._supports_prompt_caching():
            return

        # Cache the system prompt — the largest stable prefix of the request.
        if self._cache_opt("cache_system_prompt", True):
            for m in messages:
                if isinstance(m, dict) and m.get("role") == "system":
                    self._add_cache_control(m)
                    break

        # Cache through the final message for incremental multi-turn caching.
        if self._cache_opt("cache_messages", True):
            self._add_cache_control(messages[-1])

    def _maybe_cache_tools(self, tools: list) -> list:
        """
        Optionally add a ``cache_control`` breakpoint to the tool definitions.

        Tool schemas render before ``system`` in the prefix and are large and
        stable, so caching them is a big win for tool-heavy agents. Marking the
        LAST tool caches the entire tool block. Controlled by
        ``cache_config={"cache_tools": True}`` (default True); only applied for
        Anthropic-family models with ``prompt_caching`` enabled.
        """
        if not tools:
            return tools
        if not (
            self.prompt_caching and self._supports_prompt_caching()
        ):
            return tools
        if not self._cache_opt("cache_tools", True):
            return tools

        # Copy so we never mutate the shared tools_list_dictionary in place.
        cached = [
            dict(t) if isinstance(t, dict) else t for t in tools
        ]
        if isinstance(cached[-1], dict):
            cached[-1]["cache_control"] = self._cache_control_value()
        return cached

    def _apply_cache_request_params(
        self, completion_params: dict
    ) -> None:
        """
        Apply request-level caching parameters that are not message annotations.

        * OpenAI (automatic caching): pass through ``prompt_cache_key`` and
          ``prompt_cache_retention`` from ``cache_config`` when set.
        * Anthropic 1-hour TTL: attach the beta header LiteLLM needs for the
          extended cache when ``cache_config={"ttl": "1h"}``.
        """
        if not self.prompt_caching:
            return

        # OpenAI-style controls (harmless/ignored on providers that don't use
        # them; LiteLLM routes them for OpenAI-compatible backends).
        key = self._cache_opt("prompt_cache_key", None)
        if key is not None:
            completion_params["prompt_cache_key"] = key
        retention = self._cache_opt("prompt_cache_retention", None)
        if retention is not None:
            completion_params["prompt_cache_retention"] = retention

        # Anthropic 1-hour cache requires the extended-TTL beta header.
        if (
            self._supports_prompt_caching()
            and self._cache_opt("ttl", "5m") == "1h"
        ):
            headers = completion_params.get("extra_headers") or {}
            headers.setdefault(
                "anthropic-beta", "extended-cache-ttl-2025-04-11"
            )
            completion_params["extra_headers"] = headers

    # Anthropic accepts only these image formats; others fall back to JPEG.
    ANTHROPIC_IMAGE_FORMATS = frozenset(
        ["image/jpeg", "image/png", "image/gif", "image/webp"]
    )

    def _build_image_block(self, image: str) -> dict:
        """
        Build one `image_url` content block for the given image.

        Uses direct URL passing when the model supports it; otherwise converts
        the image (file path, URL, data URI, or raw base64) to a base64 data
        URI via `get_image_base64` and annotates it with its MIME type.
        """
        if self._should_use_direct_url(image):
            image_block = {
                "type": "image_url",
                "image_url": {"url": image},
            }
        else:
            # get_image_base64 always returns a data URI, so the MIME type
            # can be extracted from it directly.
            image_url = get_image_base64(image)
            mime_type = "image/jpeg"
            if "data:" in image_url and ";base64," in image_url:
                mime_type = image_url.split(";base64,")[0].split(
                    "data:"
                )[1]
            if (
                self._is_anthropic_model()
                and mime_type not in self.ANTHROPIC_IMAGE_FORMATS
            ):
                mime_type = "image/jpeg"
            image_block = {
                "type": "image_url",
                "image_url": {"url": image_url, "format": mime_type},
            }

        return image_block

    def _build_vision_message(
        self, task: str, images: Union[str, List[str]]
    ) -> dict:
        """
        Build one user message carrying the task text plus every image.

        All images ride in a single request rather than one request per image,
        which is what every vision-capable provider expects and what lets the
        model reason across the images together.
        """
        if isinstance(images, str):
            images = [images]

        return {
            "role": "user",
            "content": [{"type": "text", "text": task}]
            + [self._build_image_block(i) for i in images],
        }

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
            return _model_supports_vision(self.model_name)
        except Exception as e:
            logger.debug(
                f"Could not determine vision support for '{self.model_name}': {e}"
            )
            return False

    def vision_processing(
        self,
        task: str,
        image: Union[str, List[str]],
        messages: Optional[list] = None,
    ) -> list:
        """
        Append a vision message for the given task and image to `messages`.

        Supports file paths, HTTP/HTTPS URLs, data URIs, and raw base64
        strings. Chooses between direct URL passing and base64 conversion
        based on the image source and model capabilities; Anthropic models
        get their MIME type clamped to the formats Claude accepts.

        Args:
            task (str): The text task/prompt associated with the image.
            image (str): The image source in any supported format.
            messages (Optional[list]): The current message list. If None, an
                empty list is used.

        Returns:
            list: The updated messages list with the vision message appended.
        """
        if messages is None:
            messages = []

        logger.info(f"Processing image for model: {self.model_name}")
        messages.append(self._build_vision_message(task, image))
        return messages

    # Backwards-compatible aliases: both providers now share one code path.
    anthropic_vision_processing = vision_processing
    openai_vision_processing = vision_processing

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
        encoded_string = get_media_base64(audio)

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
            out = _model_supports_vision(self.model_name)

            if out is False:
                raise ValueError(
                    f"Model {self.model_name} does not support vision"
                )

    @staticmethod
    def check_if_model_name_uses_anthropic(model_name: str) -> bool:
        """
        Check if the model name indicates an Anthropic (Claude) model.

        Example:
            >>> LiteLLM.check_if_model_name_uses_anthropic("claude-3-opus")
            True
            >>> LiteLLM.check_if_model_name_uses_anthropic("gpt-4")
            False
        """
        name = (model_name or "").lower()
        return "claude" in name or "anthropic" in name

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
            with socket.create_connection(
                (host, port), timeout=timeout
            ):
                return True
        except OSError:
            return False

    def _build_completion_params(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        runtime_args: tuple = (),
        runtime_kwargs: Optional[dict] = None,
        messages: Optional[List[dict]] = None,
    ) -> dict:
        """
        Assemble the full parameter dict for a litellm completion call.

        Handles message preparation (including vision), parameter merging,
        tool/function config, reasoning/thinking constraints, and caching.

        Parameter priority order (highest to lowest):
            1. Runtime args (if a single dictionary)
            2. Runtime kwargs
            3. Init kwargs
            4. Init args (if a single dictionary)
            5. Defaults from __init__
        """
        completion_params = {
            "model": self.model_name,
            "messages": self._prepare_messages(
                task=task, img=img, imgs=imgs, messages=messages
            ),
            "stream": self.stream,
            "max_tokens": self.max_tokens,
            "caching": self.caching,
            "temperature": self.temperature,
        }

        # Only include top_p if explicitly set (not None)
        if self.top_p is not None:
            completion_params["top_p"] = self.top_p

        # Merge initialization kwargs first (lower priority), then runtime
        # kwargs (higher priority).
        if self.init_kwargs:
            completion_params.update(self.init_kwargs)
        if runtime_kwargs:
            completion_params.update(runtime_kwargs)

        if self.api_version is not None:
            completion_params["api_version"] = self.api_version

        if self.tools_list_dictionary is not None:
            completion_params.update(
                {
                    "tools": self._maybe_cache_tools(
                        self.tools_list_dictionary
                    ),
                    "tool_choice": self.tool_choice,
                    "parallel_tool_calls": self.parallel_tool_calls,
                }
            )

        if self.functions is not None:
            completion_params["functions"] = self.functions

        if self.base_url is not None:
            completion_params["base_url"] = self.base_url

        # Only when present: litellm falls back to the provider env var
        # on absence, and an explicit None would override that.
        if self.api_key is not None:
            completion_params["api_key"] = self.api_key

        if self.response_format is not None:
            completion_params["response_format"] = (
                self.response_format
            )

        if self.modalities and len(self.modalities) >= 2:
            completion_params["modalities"] = self.modalities

        if (
            self.reasoning_effort is not None
            and _model_supports_reasoning(self.model_name)
        ):
            completion_params["reasoning_effort"] = (
                self.reasoning_effort
            )
            # litellm maps reasoning_effort to thinking budget_tokens
            # (low=5000, medium=10000, high=15000) and max_tokens must
            # exceed that budget.
            self._apply_anthropic_thinking_constraints(
                completion_params, threshold=16000, target=16000
            )

        if (
            self.reasoning_enabled is True
            and self.thinking_tokens is not None
        ):
            completion_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_tokens,
            }
            # max_tokens must be greater than thinking budget_tokens.
            self._apply_anthropic_thinking_constraints(
                completion_params,
                threshold=self.thinking_tokens + 1,
                target=self.thinking_tokens + 1024,
            )

        # Apply request-level caching params (OpenAI keys, 1h TTL header)
        self._apply_cache_request_params(completion_params)

        # Merge init/runtime positional-dict args (highest priority)
        self._process_additional_args(completion_params, runtime_args)

        return completion_params

    def _apply_anthropic_thinking_constraints(
        self, completion_params: dict, threshold: int, target: int
    ) -> None:
        """
        Anthropic requires temperature=1, no top_p, and sufficient max_tokens
        when reasoning/thinking is enabled. No-op for other providers.
        """
        if not self._is_anthropic_model():
            return
        completion_params["temperature"] = 1
        completion_params.pop("top_p", None)
        if completion_params.get("max_tokens", 0) < threshold:
            completion_params["max_tokens"] = target

    def _process_response(self, response: any):
        """
        Route a completion response to the right output handler based on
        streaming, tools, reasoning, return_all, and model type.
        """
        if not response:
            logger.error(
                "Received empty response from completion call"
            )
            return None

        # Streaming: return the generator directly.
        if self.stream:
            return response

        # Tool calls are checked before the reasoning branch: reasoning
        # models still emit tool_calls, and routing them to
        # output_for_reasoning would drop the call.
        if self.tools_list_dictionary is not None and getattr(
            response.choices[0].message, "tool_calls", None
        ):
            return self.output_for_tools(response)

        if (
            self.reasoning_enabled
            or self.reasoning_effort is not None
            or self.thinking_tokens is not None
        ):
            return self.output_for_reasoning(response)

        if self.tools_list_dictionary is not None:
            return self.output_for_tools(response)
        if self.return_all is True:
            return response.model_dump()
        if "gemini" in self.model_name.lower():
            return gemini_output_img_handler(response)
        return response.choices[0].message.content

    def _raise_network_error(self, network_error: Exception):
        """
        Convert a low-level network exception into a NetworkConnectionError
        with a troubleshooting message tailored to the failure mode.
        """
        if not self.check_internet_connection():
            error_msg = (
                f"No internet connection detected while trying to use model '{self.model_name}'.\n\n"
                "Check your connection, or use a local model instead (e.g., Ollama):\n"
                "  model = LiteLLM(model_name='ollama/llama2')\n"
            )
        else:
            error_msg = (
                f"Network error occurred while connecting to '{self.model_name}': {network_error}\n\n"
                "The endpoint may be temporarily unavailable, the connection timed out,\n"
                "or a firewall/proxy is blocking it. Consider a local model as a fallback:\n"
                "  model = LiteLLM(model_name='ollama/llama2')\n"
            )
        logger.error(error_msg)
        raise NetworkConnectionError(error_msg) from network_error

    _NETWORK_ERRORS = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.RequestException,
        ConnectionError,
        TimeoutError,
    )

    def run(
        self,
        task: Optional[str] = None,
        audio: Optional[str] = None,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        messages: Optional[List[dict]] = None,
        *args,
        **kwargs,
    ):
        """
        Run the LLM model for the given task with optional multimodal inputs.

        Args:
            task (str): The text task or prompt to send to the model.
            audio (Optional[str]): Path or URL of an audio input, for
                audio-capable models.
            img (Optional[str]): Image file path, URL, data URI, or base64
                string, for vision-capable models.
            *args: If a single dictionary is passed, it is merged into the
                completion parameters with highest priority.
            **kwargs: Runtime parameter overrides (e.g. temperature,
                max_tokens); they override init kwargs.

        Returns:
            str | Generator | dict | list: Text content for standard
            responses; a generator when stream=True; tool-call structures
            when tools are configured; the full response dict when
            return_all=True; a file path for Gemini image generation.

        Raises:
            NetworkConnectionError: On connectivity failures, with
                troubleshooting guidance.
            ValueError: If an image is provided to a non-vision model.

        Example:
            ```python
            llm = LiteLLM(model_name="gpt-4")
            llm.run("Explain quantum computing")
            llm.run("Describe this image", img="photo.jpg")
            llm.run("Write a story", temperature=0.9, max_tokens=2000)
            ```
        """
        try:
            completion_params = self._build_completion_params(
                task,
                img=img,
                imgs=imgs,
                runtime_args=args,
                runtime_kwargs=kwargs,
                messages=messages,
            )
            response = completion(**completion_params)
            return self._process_response(response)
        except self._NETWORK_ERRORS as network_error:
            self._raise_network_error(network_error)
        except LiteLLMException as error:
            logger.error(
                f"Error in LiteLLM run: {error} Traceback: {traceback.format_exc()}"
            )
            raise
        except Exception as error:
            logger.error(
                f"Unexpected error in LiteLLM run: {error} Traceback: {traceback.format_exc()}"
            )
            raise

    async def arun(
        self,
        task: Optional[str] = None,
        audio: Optional[str] = None,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        messages: Optional[List[dict]] = None,
        *args,
        **kwargs,
    ):
        """
        Async version of `run`, using litellm's acompletion.

        Accepts the same arguments and returns the same output types as
        `run` (see its docstring).
        """
        try:
            completion_params = self._build_completion_params(
                task,
                img=img,
                imgs=imgs,
                runtime_args=args,
                runtime_kwargs=kwargs,
                messages=messages,
            )
            response = await acompletion(**completion_params)
            return self._process_response(response)
        except self._NETWORK_ERRORS as network_error:
            self._raise_network_error(network_error)
        except LiteLLMException as error:
            logger.error(
                f"Error in LiteLLM arun: {error} Traceback: {traceback.format_exc()}"
            )
            raise
        except Exception as error:
            logger.error(
                f"Unexpected error in LiteLLM arun: {error} Traceback: {traceback.format_exc()}"
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

        Tasks are processed concurrently within each batch (via a thread pool), and
        batches run one after another — useful for managing API rate limits.

        Args:
            tasks (List[str]): List of text tasks/prompts to process.
            batch_size (int): Number of tasks to process concurrently per batch. Defaults to 10.

        Returns:
            List[str]: Responses in the same order as the input tasks.

        Example:
            ```python
            llm = LiteLLM(model_name="gpt-4")
            responses = llm.batched_run(["Task 1", "Task 2", "Task 3"], batch_size=2)
            ```
        """
        # Imported here, not at module scope: swarms.structs pulls this
        # module back in, and a top-level import would be circular.
        from swarms.structs.execution_utils import run_concurrently

        return run_concurrently(
            self.run, tasks, max_workers=batch_size
        )
