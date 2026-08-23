"""
LLM manager for Swarms agents.

This module owns every piece of language-model behaviour that used to be
inlined in ``swarms/structs/agent.py``:

* construction — building the :class:`LiteLLM` instance from the agent's
  configuration, tools, and MCP tool schemas
* model rotation — the primary/fallback model list, the current index, and
  switching to the next model when one fails
* capability checks — vision, function calling, parallel function calling
* invocation — ``call()`` handles detailed streaming, panel streaming, silent
  streaming, and non-streaming calls, plus multimodal input
* stream plumbing — assembling tool-call fragments and surfacing reasoning
  ("thinking") chunks
* fallback execution — re-running a failed task down the fallback chain

Configuration stays on the ``Agent``. The manager holds a reference to its
owner and reads config live rather than snapshotting it, because agents mutate
their own config at run time — ``system_prompt`` grows when skills load,
``tools_list_dictionary`` changes when tools are registered, ``streaming_on``
is toggled per call by ``run_stream``. Anything the manager writes
(``model_name``, ``current_model_index``, ``llm``) is written back to the agent,
so serialization, ``save``/``load``, and every existing ``agent.llm`` reference
keep working unchanged.
"""

import itertools
import random
import time
import traceback
from typing import Any, Callable, List, Optional, Union

from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    InternalServerError,
)
from litellm.utils import (
    supports_function_calling,
    supports_parallel_function_calling,
    supports_vision,
)
from loguru import logger

from swarms.schemas.agent_errors import AgentLLMInitializationError
from swarms.utils.formatter import formatter
from swarms.utils.index import exists
from swarms.utils.litellm_wrapper import LiteLLM

DEFAULT_MODEL_NAME = "gpt-5.4"


class LLMManager:
    """
    Owns model selection, LiteLLM construction, and LLM invocation for an agent.

    Args:
        agent: The owning ``Agent``. Read for configuration, written to for
            ``model_name``, ``current_model_index``, and ``llm``.

    Example:
        >>> agent.llm_manager.get_current_model()
        'gpt-5.4'
        >>> agent.llm_manager.call("What is Python?", current_loop=1)
    """

    def __init__(self, agent: Any):
        self.agent = agent

    ########################################################
    # Model selection & fallback rotation
    ########################################################

    def get_available_models(self) -> List[str]:
        """
        Get the list of available models including primary and fallback models.

        Returns:
            List[str]: List of model names in order of preference
        """
        agent = self.agent
        models = []

        # If fallback_models is specified, use it as the primary list
        if agent.fallback_models:
            models = agent.fallback_models.copy()
        else:
            # Otherwise, build the list from individual parameters
            if agent.model_name:
                models.append(agent.model_name)
            if (
                agent.fallback_model_name
                and agent.fallback_model_name not in models
            ):
                models.append(agent.fallback_model_name)

        return models

    def get_current_model(self) -> str:
        """
        Get the current model being used.

        Returns:
            str: Current model name
        """
        available_models = self.get_available_models()

        if self.agent.current_model_index < len(available_models):
            return available_models[self.agent.current_model_index]

        return (
            available_models[0]
            if available_models
            else DEFAULT_MODEL_NAME
        )

    def switch_to_next_model(self) -> bool:
        """
        Switch to the next available model in the fallback list.

        Rebuilds the agent's LiteLLM instance against the new model.

        Returns:
            bool: True if successfully switched, False if models are exhausted
        """
        agent = self.agent
        available_models = self.get_available_models()

        if agent.current_model_index + 1 >= len(available_models):
            logger.error(
                f"No more models: agent '{agent.agent_name}' has exhausted all available models. "
                f"Tried {len(available_models)} models: {available_models}"
            )
            return False

        previous_model = (
            available_models[agent.current_model_index]
            if agent.current_model_index < len(available_models)
            else "Unknown"
        )
        agent.current_model_index += 1
        new_model = available_models[agent.current_model_index]

        # always log model switches
        logger.warning(
            f"[Model Switch] agent '{agent.agent_name}' switching from '{previous_model}' to fallback model: '{new_model}' "
            f"(attempt {agent.current_model_index + 1}/{len(available_models)})"
        )

        # Update the model name and reinitialize LLM
        agent.model_name = new_model
        agent.llm = self.build()

        return True

    def reset_model_index(self) -> None:
        """Reset the model index to use the primary model."""
        agent = self.agent
        agent.current_model_index = 0
        available_models = self.get_available_models()

        if available_models:
            agent.model_name = available_models[0]
            agent.llm = self.build()

    def is_fallback_available(self) -> bool:
        """
        Check if fallback models are available.

        Returns:
            bool: True if fallback models are configured
        """
        return len(self.get_available_models()) > 1

    ########################################################
    # Construction & configuration
    ########################################################

    def build(self, *args, **kwargs) -> Optional[LiteLLM]:
        """
        Build the LiteLLM instance from every configuration source.

        Combines the agent's core settings, ``llm_args``,
        ``tools_list_dictionary``, MCP tool schemas, and any arguments passed
        here into a single unified configuration.

        Args:
            *args: Additional configuration. A single dict is merged directly;
                anything else is stored under ``additional_args``.
            **kwargs: Merged into the LiteLLM configuration, taking precedence.

        Returns:
            LiteLLM: The initialized instance, or None if initialization failed.
        """
        agent = self.agent

        if agent.model_name is None:
            agent.model_name = DEFAULT_MODEL_NAME

        # Use current model (which may be a fallback) only if fallbacks are configured
        if agent.fallback_models:
            current_model = self.get_current_model()
        else:
            current_model = agent.model_name

        # Determine if parallel tool calls should be enabled
        if exists(agent.tools) and len(agent.tools) >= 2:
            parallel_tool_calls = True
        elif agent.mcp_enabled:
            parallel_tool_calls = True
        else:
            parallel_tool_calls = False

        try:
            # Base configuration that's always included
            common_args = {
                "model_name": current_model,
                "temperature": agent.temperature,
                "max_tokens": agent.max_tokens,
                "system_prompt": agent.system_prompt,
                "stream": agent.streaming_on,
                "top_p": agent.top_p,
                "retries": agent.retry_attempts,
                "reasoning_effort": agent.reasoning_effort,
                "thinking_tokens": agent.thinking_tokens,
                "reasoning_enabled": agent.reasoning_enabled,
                "agent_name": agent.agent_name,
                "prompt_caching": agent.prompt_caching,
                "cache_config": agent.cache_config,
                # Omitting these sends custom-endpoint traffic to the default
                # provider instead. temp_llm_instance_for_tool_summary
                # already forwards both.
                "base_url": agent.llm_base_url,
                "api_key": agent.llm_api_key,
            }

            # With dynamic tools, MCP schemas go into the searchable catalog
            # instead of being appended to every request. This runs before the
            # tool list is read, because deferring updates
            # `tools_list_dictionary` in place. It is a no-op without a loader,
            # and it fetches only once per agent.
            deferred_mcp = False
            if (
                agent.mcp_enabled
                and getattr(agent, "tool_loader", None) is not None
            ):
                agent.defer_mcp_tools()
                deferred_mcp = True

            # Initialize tools_list_dictionary, if applicable
            tools_list = []

            # Append tools from different sources
            if agent.tools_list_dictionary is not None:
                tools_list.extend(agent.tools_list_dictionary)

            if agent.mcp_enabled and not deferred_mcp:
                if agent.verbose:
                    logger.info(
                        f"Adding MCP tools to memory for {agent.agent_name}"
                    )

                mcp_tools = agent.add_mcp_tools_to_memory()

                if agent.verbose:
                    logger.info(f"MCP tools: {mcp_tools}")

                tools_list.extend(mcp_tools)

            # Additional arguments for LiteLLM initialization
            additional_args = {}

            if agent.llm_args is not None:
                additional_args.update(agent.llm_args)

            if tools_list:
                additional_args.update(
                    {
                        "tools_list_dictionary": tools_list,
                        "tool_choice": "auto",
                        "parallel_tool_calls": parallel_tool_calls,
                    }
                )

            if agent.mcp_enabled:
                additional_args.update({"mcp_call": True})

            # if args or kwargs are provided, then update the additional_args
            if args or kwargs:
                # Handle keyword arguments first
                if kwargs:
                    additional_args.update(kwargs)

                # Handle positional arguments (args)
                # These could be additional configuration parameters
                # that should be merged into the additional_args
                if args:
                    # If args contains a dictionary, merge it directly
                    if len(args) == 1 and isinstance(args[0], dict):
                        additional_args.update(args[0])
                    else:
                        # For other types of args, log them for debugging
                        # and potentially handle them based on their type
                        logger.debug(
                            f"Received positional args in llm_handling: {args}"
                        )
                        additional_args.update(
                            {"additional_args": args}
                        )

            # Final LiteLLM initialization with combined arguments
            return LiteLLM(**{**common_args, **additional_args})
        except AgentLLMInitializationError as e:
            logger.error(
                f"AgentLLMInitializationError: Agent Name: {agent.agent_name} Error in llm_handling: {e} "
                f"Your current configuration is not supported. Please check the configuration and parameters. "
                f"Traceback: {traceback.format_exc()}"
            )
            return None

    def get_parameters(self) -> str:
        """Return the current LiteLLM instance's attributes as a string."""
        return str(vars(self.agent.llm))

    def check_model_supports_utilities(
        self, img: Optional[str] = None
    ) -> None:
        """
        Log an error for each capability the current model is missing.

        Checks vision support when an image is supplied, function calling when
        a tool schema is set, and parallel function calling when more than two
        tools are registered. Logging only — never raises.

        Args:
            img (str, optional): Image input to check vision support for.
        """
        agent = self.agent

        # Only check vision support if an image is provided
        if img is not None:
            out = supports_vision(agent.model_name)
            if out is False:
                logger.error(
                    f"[Agent: {agent.agent_name}] Model '{agent.model_name}' does not support vision capabilities. "
                    f"Image input was provided: {img[:100]}{'...' if len(img) > 100 else ''}. "
                    f"Please use a vision-enabled model."
                )

        if agent.tools_list_dictionary is not None:
            out = supports_function_calling(agent.model_name)
            if out is False:
                logger.error(
                    f"[Agent: {agent.agent_name}] Model '{agent.model_name}' does not support function calling capabilities. "
                    f"tools_list_dictionary is set: {agent.tools_list_dictionary}. "
                    f"Please use a function calling-enabled model."
                )

        if agent.tools is not None:
            if len(agent.tools) > 2:
                out = supports_parallel_function_calling(
                    agent.model_name
                )
                if out is False:
                    logger.error(
                        f"[Agent: {agent.agent_name}] Model '{agent.model_name}' does not support parallel function calling capabilities. "
                        f"Please use a parallel function calling-enabled model."
                    )

        return None

    def randomize_temperature(self) -> None:
        """
        Randomly reset the LLM's temperature on a 0.0–1.0 scale.

        Used between loops when ``dynamic_temperature_enabled`` is set. Falls
        back to 0.5 when the LLM exposes no temperature.
        """
        try:
            if hasattr(self.agent.llm, "temperature"):
                # Randomly change the temperature attribute of the llm object
                self.agent.llm.temperature = random.uniform(0.0, 1.0)
            else:
                # Use a default temperature
                self.agent.llm.temperature = 0.5
        except Exception as error:
            logger.error(
                f"Error dynamically changing temperature: {error}"
            )

    ########################################################
    # Stream plumbing
    ########################################################

    def stream_with_tool_collection(
        self, stream, tool_calls_out: list
    ):
        """Yield every chunk unchanged while assembling delta.tool_calls fragments.

        Chunks are always forwarded so callers (print_streaming_panel, callback loops,
        etc.) can handle content deltas as normal.  Tool-call-only chunks carry no
        delta.content so existing display code skips them harmlessly.

        After the generator is fully consumed, tool_calls_out is populated with the
        assembled tool call list in the same format returned by output_for_tools, so
        the auto loop can handle it without any changes.
        """
        accumulator: dict = {}

        for chunk in stream:
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                for tc in getattr(delta, "tool_calls", None) or []:
                    idx = tc.index
                    if idx not in accumulator:
                        accumulator[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc.id:
                        accumulator[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            accumulator[idx]["function"][
                                "name"
                            ] += tc.function.name
                        if tc.function.arguments:
                            accumulator[idx]["function"][
                                "arguments"
                            ] += tc.function.arguments
            yield chunk  # always forward; callers filter on delta.content

        # Populate the output list once the stream is exhausted
        tool_calls_out.extend(
            accumulator[i] for i in sorted(accumulator)
        )

    def extract_thinking_from_stream(self, stream):
        """Yield only content chunks from a stream, displaying thinking chunks as a panel first.

        Anthropic (and other reasoning providers) emit thinking deltas before content
        deltas. This generator consumes all thinking chunks silently, then flushes a
        single thinking panel to the console, and finally yields every subsequent
        content chunk unchanged so callers can handle them normally.
        """
        thinking_parts = []
        thinking_displayed = False

        for chunk in stream:
            if not hasattr(chunk, "choices") or not chunk.choices:
                yield chunk
                continue

            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)

            if reasoning:
                thinking_parts.append(reasoning)
                continue  # swallow the thinking chunk; don't pass to content stream

            # First non-thinking chunk — flush accumulated thinking
            if thinking_parts and not thinking_displayed:
                if self.agent.print_on:
                    formatter.print_thinking_panel(
                        "".join(thinking_parts),
                        title=self._thinking_title(),
                    )
                thinking_displayed = True

            yield chunk

        # Edge case: stream ended with only thinking and no content chunks
        if (
            thinking_parts
            and not thinking_displayed
            and self.agent.print_on
        ):
            formatter.print_thinking_panel(
                "".join(thinking_parts), title=self._thinking_title()
            )

    def _thinking_title(self) -> str:
        """Panel title for reasoning output."""
        return (
            f"{self.agent.agent_name} | Thinking"
            if self.agent.agent_name
            else "Thinking"
        )

    ########################################################
    # Invocation
    ########################################################

    def call(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        current_loop: int = 0,
        streaming_callback: Optional[Callable[[str], None]] = None,
        *args,
        **kwargs,
    ) -> Any:
        """
        Call the LLM, handling streaming and multimodal inputs.

        Three modes of operation, selected from the agent's config:

        1. **Detailed streaming** (``agent.stream``): streams tokens with full
           metadata (citations, usage, logprobs, …), passing a ``token_info``
           dict to ``streaming_callback`` per token.
        2. **Panel streaming** (``agent.streaming_on``): streams with formatted
           panels, a real-time callback, or silently when ``print_on`` is False.
        3. **Non-streaming**: a direct ``llm.run()`` returning the full string.

        Args:
            task (str): The task or prompt to send to the LLM.
            img (Optional[str]): Image input for multimodal processing — a file
                path, URL, data URI, or raw base64 string.
            current_loop (int): Loop iteration, used in streaming panel titles.
            streaming_callback (Optional[Callable[[str], None]]): Receives
                ``token_info`` dicts in detailed streaming, token strings in
                panel streaming.
            *args: Passed through to ``llm.run()``.
            **kwargs: Passed through to ``llm.run()``. ``is_last`` is filtered out.

        Returns:
            Any: The complete response string, or the assembled tool-call list
                when the model made tool calls mid-stream.

        Raises:
            AgentLLMError, BadRequestError, InternalServerError,
            AuthenticationError, Exception: re-raised for upstream handling.
        """
        agent = self.agent

        # Filter out is_last from kwargs if present
        if "is_last" in kwargs:
            del kwargs["is_last"]

        try:
            if agent.stream and hasattr(agent.llm, "stream"):
                return self._call_detailed_streaming(
                    task, img, streaming_callback, *args, **kwargs
                )

            if agent.streaming_on and hasattr(agent.llm, "stream"):
                return self._call_panel_streaming(
                    task,
                    img,
                    current_loop,
                    streaming_callback,
                    *args,
                    **kwargs,
                )

            run_args = {"task": task}

            if img is not None:
                run_args["img"] = img
            if imgs:
                run_args["imgs"] = imgs

            return agent.llm.run(**run_args, **kwargs)

        except (
            BadRequestError,
            InternalServerError,
            AuthenticationError,
            Exception,
        ) as e:
            logger.error(
                f"Error calling LLM with model '{self.get_current_model()}': {e}. "
                f"Task: {task}, Args: {args}, Kwargs: {kwargs} Traceback: {traceback.format_exc()}"
            )
            raise e

    def _run_stream(
        self, task: str, img: Optional[str], *args, **kwargs
    ):
        """Invoke ``llm.run`` with streaming forced on, returning (response, original_stream)."""
        original_stream = self.agent.llm.stream
        self.agent.llm.stream = True

        if img is not None:
            response = self.agent.llm.run(
                task=task, img=img, *args, **kwargs
            )
        else:
            response = self.agent.llm.run(task=task, *args, **kwargs)

        return response, original_stream

    def _call_detailed_streaming(
        self,
        task: str,
        img: Optional[str],
        streaming_callback: Optional[Callable[[str], None]],
        *args,
        **kwargs,
    ) -> Any:
        """Stream tokens with full per-token metadata printed and forwarded."""
        streaming_response, original_stream = self._run_stream(
            task, img, *args, **kwargs
        )

        if not hasattr(streaming_response, "__iter__") or isinstance(
            streaming_response, str
        ):
            self.agent.llm.stream = original_stream
            return streaming_response

        complete_response = ""
        token_count = 0
        final_chunk = None
        first_chunk = None
        tool_calls_out: list = []

        for chunk in self.stream_with_tool_collection(
            self.extract_thinking_from_stream(streaming_response),
            tool_calls_out,
        ):
            if first_chunk is None:
                first_chunk = chunk

            if (
                hasattr(chunk, "choices")
                and chunk.choices[0].delta.content
            ):
                content = chunk.choices[0].delta.content
                complete_response += content
                token_count += 1

                token_info = self._build_token_info(
                    chunk, content, token_count
                )

                print(f"ResponseStream {token_info}")

                if streaming_callback is not None:
                    streaming_callback(token_info)

            final_chunk = chunk

        print(self._format_final_chunk(final_chunk))

        self.agent.llm.stream = original_stream

        return tool_calls_out if tool_calls_out else complete_response

    def _build_token_info(
        self, chunk: Any, content: str, token_count: int
    ) -> dict:
        """Schema emitted per token during detailed streaming."""
        return {
            "token_index": token_count,
            "model": getattr(
                chunk, "model", self.get_current_model()
            ),
            "id": getattr(chunk, "id", ""),
            "created": getattr(chunk, "created", int(time.time())),
            "object": getattr(
                chunk, "object", "chat.completion.chunk"
            ),
            "token": content,
            "system_fingerprint": getattr(
                chunk, "system_fingerprint", ""
            ),
            "finish_reason": chunk.choices[0].finish_reason,
            "citations": getattr(chunk, "citations", None),
            "provider_specific_fields": getattr(
                chunk, "provider_specific_fields", None
            ),
            "service_tier": getattr(chunk, "service_tier", "default"),
            "obfuscation": getattr(chunk, "obfuscation", None),
            "usage": getattr(chunk, "usage", None),
            "logprobs": chunk.choices[0].logprobs,
            "timestamp": time.time(),
        }

    def _format_final_chunk(self, final_chunk: Any) -> str:
        """Render the trailing ModelResponseStream summary line."""
        header = (
            f"ModelResponseStream(id='{getattr(final_chunk, 'id', 'N/A')}', "
            f"created={getattr(final_chunk, 'created', 'N/A')}, "
            f"model='{getattr(final_chunk, 'model', self.get_current_model())}', "
            f"object='{getattr(final_chunk, 'object', 'chat.completion.chunk')}', "
            f"system_fingerprint='{getattr(final_chunk, 'system_fingerprint', 'N/A')}', "
            f"choices=[StreamingChoices(finish_reason='{final_chunk.choices[0].finish_reason}', "
            f"index=0, delta=Delta(provider_specific_fields=None, content=None, role=None, "
            f"function_call=None, tool_calls=None, audio=None), logprobs=None)], "
            f"provider_specific_fields=None"
        )

        if not (
            final_chunk
            and hasattr(final_chunk, "usage")
            and final_chunk.usage
        ):
            return f"{header})"

        usage = final_chunk.usage

        return (
            f"{header}, "
            f"usage=Usage(completion_tokens={usage.completion_tokens}, "
            f"prompt_tokens={usage.prompt_tokens}, "
            f"total_tokens={usage.total_tokens}, "
            f"completion_tokens_details=CompletionTokensDetailsWrapper("
            f"accepted_prediction_tokens={usage.completion_tokens_details.accepted_prediction_tokens}, "
            f"audio_tokens={usage.completion_tokens_details.audio_tokens}, "
            f"reasoning_tokens={usage.completion_tokens_details.reasoning_tokens}, "
            f"rejected_prediction_tokens={usage.completion_tokens_details.rejected_prediction_tokens}, "
            f"text_tokens={usage.completion_tokens_details.text_tokens}), "
            f"prompt_tokens_details=PromptTokensDetailsWrapper("
            f"audio_tokens={usage.prompt_tokens_details.audio_tokens}, "
            f"cached_tokens={usage.prompt_tokens_details.cached_tokens}, "
            f"text_tokens={usage.prompt_tokens_details.text_tokens}, "
            f"image_tokens={usage.prompt_tokens_details.image_tokens})))"
        )

    def _call_panel_streaming(
        self,
        task: str,
        img: Optional[str],
        current_loop: int,
        streaming_callback: Optional[Callable[[str], None]],
        *args,
        **kwargs,
    ) -> Any:
        """Stream with a callback, silently, or through a Rich streaming panel."""
        streaming_response, original_stream = self._run_stream(
            task, img, *args, **kwargs
        )

        if not hasattr(streaming_response, "__iter__") or isinstance(
            streaming_response, str
        ):
            # Restore original stream setting
            self.agent.llm.stream = original_stream
            return streaming_response

        tool_calls_out: list = []

        # Check if streaming_callback is provided (for ConcurrentWorkflow dashboard integration)
        if streaming_callback is not None:
            # Real-time callback streaming — thinking panel is printed as a
            # side effect inside extract_thinking_from_stream; no Live context
            # is active so there is no interleaving risk.
            complete_response = self._collect_stream(
                self.stream_with_tool_collection(
                    self.extract_thinking_from_stream(
                        streaming_response
                    ),
                    tool_calls_out,
                ),
                streaming_callback,
            )
        elif self.agent.print_on is False:
            # Silent streaming - no printing, just collect chunks
            complete_response = self._collect_stream(
                self.stream_with_tool_collection(
                    streaming_response, tool_calls_out
                )
            )
        else:
            complete_response = self._stream_to_panel(
                streaming_response, tool_calls_out, current_loop
            )

        # Restore original stream setting
        self.agent.llm.stream = original_stream

        # If the model made tool calls during the stream, return them
        # so the auto loop executes them — same format as non-streaming.
        return tool_calls_out if tool_calls_out else complete_response

    def _collect_stream(
        self,
        stream,
        streaming_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Drain a stream into a string, optionally forwarding each token."""
        chunks = []

        for chunk in stream:
            if (
                hasattr(chunk, "choices")
                and chunk.choices[0].delta.content
            ):
                content = chunk.choices[0].delta.content
                chunks.append(content)

                if streaming_callback is not None:
                    streaming_callback(content)

        return "".join(chunks)

    def _stream_to_panel(
        self,
        streaming_response,
        tool_calls_out: list,
        current_loop: int,
    ) -> str:
        """Display a live streaming panel and return the collected response.

        The print-streaming path uses Rich's Live display. We MUST print the
        thinking panel BEFORE the Live context opens, otherwise the
        console.print() call inside extract_thinking_from_stream interleaves
        with the Live display and corrupts the terminal.
        """
        thinking_parts: list = []
        first_content_chunk = None

        for chunk in streaming_response:
            delta = (
                chunk.choices[0].delta
                if hasattr(chunk, "choices") and chunk.choices
                else None
            )
            reasoning = (
                getattr(delta, "reasoning_content", None)
                if delta
                else None
            )
            if reasoning:
                thinking_parts.append(reasoning)
            else:
                first_content_chunk = chunk
                break

        if thinking_parts:
            formatter.print_thinking_panel(
                "".join(thinking_parts), title=self._thinking_title()
            )

        # Chain the already-consumed first chunk with the remaining stream,
        # then wrap with tool-call collection.
        chained = itertools.chain(
            (
                [first_content_chunk]
                if first_content_chunk is not None
                else []
            ),
            streaming_response,
        )

        # Use the streaming panel to display and collect the response
        return formatter.print_streaming_panel(
            self.stream_with_tool_collection(chained, tool_calls_out),
            title=f"🤖 Agent: {self.agent.agent_name} Loops: {current_loop}",
            style=None,  # Use random color like non-streaming approach
            collect_chunks=True,
        )

    ########################################################
    # Fallback execution
    ########################################################

    def handle_fallback_execution(
        self,
        task: Optional[Union[str, Any]] = None,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        correct_answer: Optional[str] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
        original_error: Exception = None,
        *args,
        **kwargs,
    ) -> Any:
        """
        Re-run a failed task against the next fallback model.

        Recurses down the fallback chain until the task succeeds or every model
        is exhausted, at which point the original error is handed to the
        agent's error handler.

        Args:
            task: The task to be executed.
            img: The image to be processed.
            imgs: The list of images to be processed.
            correct_answer: The correct answer for continuous run mode.
            streaming_callback: Callback receiving streaming tokens in real time.
            original_error (Exception): The error that triggered the fallback.
            *args: Passed through to the agent's ``run``.
            **kwargs: Passed through to the agent's ``run``.

        Returns:
            Any: The result of a successful run, or None once models run out.
        """
        agent = self.agent

        # Check if fallback models are available
        if not self.is_fallback_available():
            if agent.verbose:
                logger.error(
                    f"Agent Name: {agent.agent_name} [NO FALLBACK] failed with model '{self.get_current_model()}' "
                    f"and no fallback models are configured. Error: {str(original_error)[:100]}{'...' if len(str(original_error)) > 100 else ''}"
                )
            agent._handle_run_error(original_error)
            return None

        # Try to switch to the next fallback model
        if not self.switch_to_next_model():
            if agent.verbose:
                logger.error(
                    f"Agent Name: {agent.agent_name} [FALLBACK EXHAUSTED] has exhausted all available models. "
                    f"Tried {len(self.get_available_models())} models: {self.get_available_models()}"
                )
            agent._handle_run_error(original_error)
            return None

        # Log fallback attempt
        if agent.verbose:
            logger.warning(
                f"Agent Name: {agent.agent_name} [FALLBACK] failed with model '{self.get_current_model()}'. "
                f"Switching to fallback model '{self.get_current_model()}' (attempt {agent.current_model_index + 1}/{len(self.get_available_models())})"
            )

        try:
            # Recursive call to run() with the new model
            result = agent.run(
                task=task,
                img=img,
                imgs=imgs,
                correct_answer=correct_answer,
                streaming_callback=streaming_callback,
                *args,
                **kwargs,
            )

            if agent.verbose:
                # Log successful completion with fallback model
                logger.info(
                    f"Agent Name: {agent.agent_name} [FALLBACK SUCCESS] successfully completed task "
                    f"using fallback model '{self.get_current_model()}'"
                )

            return result

        except Exception as fallback_error:
            logger.error(
                f"Agent Name: {agent.agent_name} Fallback model '{self.get_current_model()}' also failed: {fallback_error}"
            )

            # Try the next fallback model recursively
            return self.handle_fallback_execution(
                task=task,
                img=img,
                imgs=imgs,
                correct_answer=correct_answer,
                streaming_callback=streaming_callback,
                original_error=original_error,
                *args,
                **kwargs,
            )
