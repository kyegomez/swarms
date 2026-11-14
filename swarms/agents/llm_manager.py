"""
LLM manager for Swarms agents.

This module owns every piece of language-model behaviour that used to be
inlined in ``swarms/structs/agent.py``:

* construction — building the :class:`LiteLLM` instance from the agent's
  configuration, tools, and MCP tool schemas
* capability checks — vision, function calling, parallel function calling
* invocation — ``call()`` handles detailed streaming, panel streaming, silent
  streaming, and non-streaming calls, plus multimodal input
* stream plumbing — assembling tool-call fragments and surfacing reasoning
  ("thinking") chunks

Configuration stays on the ``Agent``. The manager holds a reference to its
owner and reads config live rather than snapshotting it, because agents mutate
their own config at run time — ``tools_list_dictionary`` changes when tools
are registered, ``streaming_on`` is toggled per call by ``run_stream``.
The ``llm`` it builds is written back to the agent, so serialization,
``save``/``load``, and every existing ``agent.llm`` reference keep working
unchanged.
"""

import asyncio
import inspect
import random
import time
import traceback
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Callable, List, Optional

from loguru import logger

from swarms.schemas.agent_errors import AgentLLMInitializationError
from swarms.telemetry.otel import swarm_telemetry
from swarms.utils import llm_backend
from swarms.utils.formatter import formatter
from swarms.utils.index import exists
from swarms.utils.litellm_wrapper import LiteLLM

DEFAULT_MODEL_NAME = "gpt-5.4"

# Messages already logged, so a capability gap is reported once per process.
_logged_capability_messages = set()


@lru_cache(maxsize=None)
def _model_supports_function_calling(model_name: str) -> bool:
    """Whether the model supports function calling.

    Args:
        model_name: The model to look up.

    Returns:
        bool: The lookup result.
    """
    return llm_backend.supports_function_calling(model_name)


@lru_cache(maxsize=None)
def _model_supports_parallel_function_calling(
    model_name: str,
) -> bool:
    """Whether the model supports parallel function calling.

    Args:
        model_name: The model to look up.

    Returns:
        bool: The lookup result.
    """
    return llm_backend.supports_parallel_function_calling(model_name)


def _log_once(level: str, message: str) -> None:
    """Log a message the first time it occurs in this process.

    Args:
        level: The log level name.
        message: The message to log.
    """
    if message in _logged_capability_messages:
        return
    _logged_capability_messages.add(message)
    logger.log(level, message)


def _collect_tool_call_deltas(accumulator: dict, chunk: Any) -> None:
    """Fold one chunk's tool call fragments into the calls assembled so far.

    Args:
        accumulator: Tool calls keyed by their index in the response.
        chunk: A chunk from the model stream.
    """
    if not (hasattr(chunk, "choices") and chunk.choices):
        return
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


class _StreamParts:
    """Turns model stream chunks into labelled text parts, one chunk at a time."""

    def __init__(
        self,
        manager: "LLMManager",
        tool_calls_out: list,
        content_out: list,
    ):
        self.manager = manager
        self.tool_calls_out = tool_calls_out
        self.content_out = content_out
        self.accumulator: dict = {}
        # The last call of this run ended mid-line, so this one starts on a new line.
        self.section = "carry" if manager.stream_mid_line else None
        self.tool_names: dict = {}
        self.tool_args_started: set = set()

    def feed(self, chunk: Any) -> list:
        """Return the (kind, text, chunk) parts one chunk carries.

        Args:
            chunk: A chunk from the model stream.

        Returns:
            list: The parts, in the order they should be shown.
        """
        _collect_tool_call_deltas(self.accumulator, chunk)
        # RouteHub forwards chunks with no choices, which litellm drops.
        if not getattr(chunk, "choices", None):
            return []
        parts = []
        delta = chunk.choices[0].delta

        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            if self.section != "reasoning":
                gap = "" if self.section is None else "\n"
                parts.append(
                    ("reasoning", f"{gap}[thinking] ", chunk)
                )
                self.section = "reasoning"
            parts.append(("reasoning", reasoning, chunk))

        content = getattr(delta, "content", None)
        if content:
            if self.section not in (None, "content"):
                parts.append(("content", "\n", chunk))
            self.section = "content"
            self.content_out.append(content)
            parts.append(("content", content, chunk))

        for tc in getattr(delta, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", None) or ""
            arguments = getattr(fn, "arguments", None) or ""
            if self.section != ("tool_call", tc.index):
                gap = "" if self.section is None else "\n"
                resumed = self.tool_names.get(tc.index, "")
                if tc.index in self.tool_args_started:
                    resumed += " "
                parts.append(
                    (
                        "tool_call",
                        f"{gap}[tool call] {resumed}",
                        chunk,
                    )
                )
                self.section = ("tool_call", tc.index)
            if name:
                self.tool_names[tc.index] = (
                    self.tool_names.get(tc.index, "") + name
                )
                parts.append(("tool_call", name, chunk))
            if arguments:
                if tc.index not in self.tool_args_started:
                    self.tool_args_started.add(tc.index)
                    parts.append(("tool_call", " ", chunk))
                parts.append(("tool_call", arguments, chunk))
        return parts

    def finish(self) -> list:
        """Close the stream: assemble the tool calls and end an open line.

        Returns:
            list: The closing parts, empty when the stream ended on content.
        """
        self.tool_calls_out.extend(
            self.accumulator[i] for i in sorted(self.accumulator)
        )
        parts = []
        if self.section not in (None, "content", "carry"):
            kind = (
                "reasoning"
                if self.section == "reasoning"
                else "tool_call"
            )
            parts.append((kind, "\n", None))
        self.manager.stream_mid_line = self.section in (
            "content",
            "carry",
        )
        return parts


async def emit_token(
    callback: Optional[Callable], token: Any
) -> None:
    """Send one streamed token to a callback, awaiting it when it is async.

    Args:
        callback: A sync or async function taking the token, or None.
        token: The token text, or a token_info dict in detailed streaming.
    """
    if callback is None:
        return
    result = callback(token)
    if inspect.isawaitable(result):
        await result


async def _aiter_sync(stream: Any):
    """Read a blocking chunk stream on a worker thread, one chunk per await.

    Args:
        stream: A sync iterable of chunks.

    Yields:
        Any: Each chunk, in order.
    """
    iterator = iter(stream)
    done = object()
    while True:
        chunk = await asyncio.to_thread(next, iterator, done)
        if chunk is done:
            return
        yield chunk


class LLMManager:
    """
    Owns LiteLLM construction and LLM invocation for an agent.

    Args:
        agent: The owning ``Agent``. Read for configuration, written to for
            ``llm``.

    Example:
        >>> agent.llm_manager.call("What is Python?", current_loop=1)
    """

    def __init__(self, agent: Any):
        self.agent = agent
        self.stream_mid_line = False

    # Construction and configuration.

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
                "model_name": agent.model_name,
                "temperature": agent.temperature,
                "max_tokens": agent.max_tokens,
                "system_prompt": agent.system_prompt,
                "stream": agent.streaming_on,
                "top_p": agent.top_p,
                "retries": agent.retry_attempts,
                "reasoning_effort": agent.reasoning_effort,
                "thinking_tokens": agent.thinking_tokens,
                "response_format": agent.output_schema,
                "reasoning_enabled": agent.reasoning_enabled,
                "agent_name": agent.agent_name,
                "prompt_caching": agent.prompt_caching,
                "cache_config": agent.cache_config,
                # The agent keeps the running total, so a rebuilt LLM does not reset it.
                "usage_hook": agent._add_usage,
                # Omitting these sends custom-endpoint traffic to the default provider instead.
                "base_url": agent.llm_base_url,
                "api_key": agent.llm_api_key,
            }

            # Runs before the tool list is read, because deferring updates tools_list_dictionary in place.
            deferred_mcp = False
            if (
                agent.mcp_enabled
                and getattr(agent, "tool_loader", None) is not None
            ):
                agent.tool_manager.defer_mcp_tools()
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

                mcp_tools = (
                    agent.tool_manager.add_mcp_tools_to_memory()
                )

                if agent.verbose:
                    logger.info(f"MCP tools: {mcp_tools}")

                tools_list.extend(mcp_tools)

            # Additional arguments for LiteLLM initialization
            additional_args = {}

            # The provider stops generating at the token; llm_args can still override stop.
            if agent.stopping_token:
                additional_args["stop"] = [agent.stopping_token]

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

                if args:
                    # If args contains a dictionary, merge it directly
                    if len(args) == 1 and isinstance(args[0], dict):
                        additional_args.update(args[0])
                    else:
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
        Log the capabilities the current model lacks for this run.

        Args:
            img (str, optional): Image input to check vision support for.
        """
        agent = self.agent

        # Only check vision support if an image is provided
        if img is not None:
            out = llm_backend.supports_vision(agent.model_name)
            if out is False:
                logger.error(
                    f"[Agent: {agent.agent_name}] Model '{agent.model_name}' does not support vision capabilities. "
                    f"Image input was provided: {img[:100]}{'...' if len(img) > 100 else ''}. "
                    f"Please use a vision-enabled model."
                )

        if agent.tools_list_dictionary:
            out = _model_supports_function_calling(agent.model_name)
            if out is False:
                _log_once(
                    "ERROR",
                    f"[Agent: {agent.agent_name}] Model '{agent.model_name}' does not support function calling capabilities, "
                    f"but tools are configured. "
                    f"Please use a function calling-enabled model.",
                )

        if agent.tools is not None:
            if len(agent.tools) > 2:
                out = _model_supports_parallel_function_calling(
                    agent.model_name
                )
                # litellm's capability data lags new models, so this is only a hint.
                if out is False:
                    _log_once(
                        "WARNING",
                        f"Model '{agent.model_name}' is not listed as supporting parallel function calling, "
                        f"so its tool calls may arrive one per turn.",
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

    # Stream plumbing.

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
            _collect_tool_call_deltas(accumulator, chunk)
            yield chunk  # always forward; callers filter on delta.content

        # Populate the output list once the stream is exhausted
        tool_calls_out.extend(
            accumulator[i] for i in sorted(accumulator)
        )

    def stream_parts(
        self, stream, tool_calls_out: list, content_out: list
    ):
        """Yield everything the model streams as it arrives: reasoning, content and tool calls.

        Content passes through unchanged. Reasoning starts with a [thinking] header and
        each tool call with a [tool call] header naming the tool, followed by its argument
        fragments, so a reader of the text can tell the parts apart.

        Args:
            stream: The raw chunk stream from the model.
            tool_calls_out: Filled with the assembled tool calls once the stream ends.
            content_out: Receives each content fragment; joined, they are the reply text.

        Yields:
            tuple: (kind, text, chunk), where kind is "reasoning", "content" or "tool_call".
        """
        parser = _StreamParts(self, tool_calls_out, content_out)
        for chunk in stream:
            yield from parser.feed(chunk)
        yield from parser.finish()

    async def astream_parts(
        self, stream, tool_calls_out: list, content_out: list
    ):
        """Yield everything an async stream carries as it arrives.

        Args:
            stream: The raw async chunk stream from the model.
            tool_calls_out: Filled with the assembled tool calls once the stream ends.
            content_out: Receives each content fragment; joined, they are the reply text.

        Yields:
            tuple: (kind, text, chunk), where kind is "reasoning", "content" or "tool_call".
        """
        parser = _StreamParts(self, tool_calls_out, content_out)
        async for chunk in stream:
            for part in parser.feed(chunk):
                yield part
        for part in parser.finish():
            yield part

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

    # Invocation.

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
        2. **Panel streaming** (``agent.streaming_on`` or a ``streaming_callback``):
           formatted panels, a real-time callback, or silent when ``print_on`` is False.
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
        with self._call_span(current_loop) as outcome:
            outcome["result"] = self._call_llm(
                task,
                img,
                imgs,
                current_loop,
                streaming_callback,
                *args,
                **kwargs,
            )
        return outcome["result"]

    async def acall(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        current_loop: int = 0,
        streaming_callback: Optional[Callable[[str], Any]] = None,
        *args,
        **kwargs,
    ) -> Any:
        """
        Call the LLM on the event loop, with the same three modes as call.

        Args:
            task (str): The task or prompt to send to the LLM.
            img (Optional[str]): Image input for multimodal processing.
            imgs (Optional[List[str]]): Several image inputs.
            current_loop (int): Loop iteration, used in streaming panel titles.
            streaming_callback (Optional[Callable[[str], Any]]): A sync or async
                function receiving each token as the provider sends it.
            *args: Passed through to the LLM.
            **kwargs: Passed through to the LLM.

        Returns:
            Any: The complete response string, or the assembled tool-call list.
        """
        with self._call_span(current_loop) as outcome:
            outcome["result"] = await self._acall_llm(
                task,
                img,
                imgs,
                current_loop,
                streaming_callback,
                *args,
                **kwargs,
            )
        return outcome["result"]

    @contextmanager
    def _call_span(self, current_loop: int):
        """Trace one model call; the caller stores the reply under "result".

        Args:
            current_loop (int): Loop iteration, recorded on the span.

        Yields:
            dict: Holder the caller puts the model's reply in.
        """
        agent = self.agent
        outcome: dict = {}
        with swarm_telemetry().capture_run(
            "Agent.llm_call", agent
        ) as span:
            span.set("gen_ai.operation.name", "chat")
            span.set("gen_ai.request.model", agent.model_name)
            span.set("swarms.loop", current_loop)
            usage_before = dict(getattr(agent, "_usage", None) or {})
            try:
                yield outcome
            finally:
                usage_after = getattr(agent, "_usage", None)
                if isinstance(usage_after, dict):
                    span.record_usage(
                        {
                            key: value - usage_before.get(key, 0)
                            for key, value in usage_after.items()
                        }
                    )
                request = getattr(agent.llm, "last_request", None)
                if request is not None:
                    span.record_json("swarms.request", request)
            result = outcome.get("result")
            response = getattr(agent.llm, "last_response", None)
            # A stream leaves no response object, so the assembled result stands in.
            span.record_json(
                "swarms.response",
                (
                    response.model_dump()
                    if hasattr(response, "model_dump")
                    else result if response is None else response
                ),
            )
            finish_reason = getattr(
                agent.llm, "last_finish_reason", None
            )
            if finish_reason:
                span.set(
                    "gen_ai.response.finish_reasons", finish_reason
                )
            response_model = getattr(
                agent.llm, "last_response_model", None
            )
            if response_model:
                span.set("gen_ai.response.model", response_model)
            span.record_success()

    def _call_llm(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        current_loop: int = 0,
        streaming_callback: Optional[Callable[[str], None]] = None,
        *args,
        **kwargs,
    ) -> Any:
        """Run the model call for one loop, picking the streaming mode from the agent's config.

        Args:
            task (str): The task or prompt to send to the LLM.
            img (Optional[str]): Image input for multimodal processing.
            imgs (Optional[List[str]]): Several image inputs.
            current_loop (int): Loop iteration, used in streaming panel titles.
            streaming_callback (Optional[Callable[[str], None]]): Receives streamed tokens.
            *args: Passed through to the LLM.
            **kwargs: Passed through to the LLM.

        Returns:
            Any: The complete response string, or the tool-call list.
        """
        agent = self.agent

        # Filter out is_last from kwargs if present
        if "is_last" in kwargs:
            del kwargs["is_last"]

        if imgs:
            kwargs["imgs"] = imgs

        if agent.stream and hasattr(agent.llm, "stream"):
            return self._call_detailed_streaming(
                task, img, streaming_callback, *args, **kwargs
            )

        if (
            agent.streaming_on or streaming_callback is not None
        ) and hasattr(agent.llm, "stream"):
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

        return agent.llm.run(**run_args, **kwargs)

    def _run_stream(
        self,
        task: str,
        img: Optional[str],
        consume: Callable[[Any], Any],
        *args,
        **kwargs,
    ) -> Any:
        """Invoke ``llm.run`` with streaming forced on and pass the stream to ``consume``, restoring ``llm.stream`` even if either raises."""
        llm = self.agent.llm
        original_stream = llm.stream
        llm.stream = True
        try:
            if img is not None:
                response = llm.run(
                    task=task, img=img, *args, **kwargs
                )
            else:
                response = llm.run(task=task, *args, **kwargs)

            if not hasattr(response, "__iter__") or isinstance(
                response, str
            ):
                return response

            return consume(response)
        finally:
            llm.stream = original_stream

    def _call_detailed_streaming(
        self,
        task: str,
        img: Optional[str],
        streaming_callback: Optional[Callable[[str], None]],
        *args,
        **kwargs,
    ) -> Any:
        """Stream tokens with full per-token metadata printed and forwarded."""
        return self._run_stream(
            task,
            img,
            lambda stream: self._consume_detailed(
                stream, streaming_callback
            ),
            *args,
            **kwargs,
        )

    def _consume_detailed(
        self,
        streaming_response,
        streaming_callback: Optional[Callable[[dict], None]],
    ) -> Any:
        """Print and forward a token_info dict for every streamed part.

        Args:
            streaming_response: The raw chunk stream from the model.
            streaming_callback: Receives each token_info dict, if given.

        Returns:
            Any: The assembled tool calls if the model made any, else the reply text.
        """
        token_count = 0
        last_chunk = [None]
        tool_calls_out: list = []
        content_out: list = []

        def track_last(stream):
            for chunk in stream:
                last_chunk[0] = chunk
                yield chunk

        for kind, text, chunk in self.stream_parts(
            track_last(streaming_response),
            tool_calls_out,
            content_out,
        ):
            token_count += 1
            token_info = self._build_token_info(
                chunk or last_chunk[0], text, token_count, kind
            )

            print(f"ResponseStream {token_info}")

            if streaming_callback is not None:
                streaming_callback(token_info)

        complete_response = "".join(content_out)
        print(self._format_final_chunk(last_chunk[0]))

        return tool_calls_out if tool_calls_out else complete_response

    def _build_token_info(
        self,
        chunk: Any,
        content: str,
        token_count: int,
        kind: str = "content",
    ) -> dict:
        """Schema emitted per token during detailed streaming."""
        choice = (
            chunk.choices[0]
            if getattr(chunk, "choices", None)
            else None
        )
        return {
            "token_index": token_count,
            "type": kind,
            "model": getattr(chunk, "model", self.agent.model_name),
            "id": getattr(chunk, "id", ""),
            "created": getattr(chunk, "created", int(time.time())),
            "object": getattr(
                chunk, "object", "chat.completion.chunk"
            ),
            "token": content,
            "system_fingerprint": getattr(
                chunk, "system_fingerprint", ""
            ),
            "finish_reason": getattr(choice, "finish_reason", None),
            "citations": getattr(chunk, "citations", None),
            "provider_specific_fields": getattr(
                chunk, "provider_specific_fields", None
            ),
            "service_tier": getattr(chunk, "service_tier", "default"),
            "obfuscation": getattr(chunk, "obfuscation", None),
            "usage": getattr(chunk, "usage", None),
            "logprobs": getattr(choice, "logprobs", None),
            "timestamp": time.time(),
        }

    def _format_final_chunk(self, final_chunk: Any) -> str:
        """Render the trailing ModelResponseStream summary line."""
        header = (
            f"ModelResponseStream(id='{getattr(final_chunk, 'id', 'N/A')}', "
            f"created={getattr(final_chunk, 'created', 'N/A')}, "
            f"model='{getattr(final_chunk, 'model', self.agent.model_name)}', "
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
        return self._run_stream(
            task,
            img,
            lambda stream: self._consume_panel(
                stream, current_loop, streaming_callback
            ),
            *args,
            **kwargs,
        )

    def _consume_panel(
        self,
        streaming_response,
        current_loop: int,
        streaming_callback: Optional[Callable[[str], None]],
    ) -> Any:
        """Send every streamed part to the callback, the live panel, or nowhere when printing is off.

        Args:
            streaming_response: The raw chunk stream from the model.
            current_loop: The loop number shown in the panel title.
            streaming_callback: Receives the text of every part, if given.

        Returns:
            Any: The assembled tool calls if the model made any, else the reply text.
        """
        tool_calls_out: list = []

        # A callback (e.g. a ConcurrentWorkflow dashboard) takes precedence over the panel
        if streaming_callback is not None:
            complete_response = self._collect_stream(
                streaming_response, streaming_callback, tool_calls_out
            )
        elif self.agent.print_on is False:
            complete_response = self._collect_stream(
                streaming_response, tool_calls_out=tool_calls_out
            )
        else:
            complete_response = self._stream_to_panel(
                streaming_response, tool_calls_out, current_loop
            )

        # Tool calls made mid-stream come back in the non-streaming shape
        return tool_calls_out if tool_calls_out else complete_response

    def _collect_stream(
        self,
        stream,
        streaming_callback: Optional[Callable[[str], None]] = None,
        tool_calls_out: Optional[list] = None,
    ) -> str:
        """Drain a stream into its reply text, forwarding every streamed part to the callback.

        Args:
            stream: The raw chunk stream from the model.
            streaming_callback: Receives the text of every part: reasoning, content and tool calls.
            tool_calls_out: Filled with the assembled tool calls, if given.

        Returns:
            str: The reply's content text, without reasoning or tool calls.
        """
        content_out: list = []

        for _, text, _ in self.stream_parts(
            stream,
            tool_calls_out if tool_calls_out is not None else [],
            content_out,
        ):
            if streaming_callback is not None:
                streaming_callback(text)

        return "".join(content_out)

    def _stream_to_panel(
        self,
        streaming_response,
        tool_calls_out: list,
        current_loop: int,
    ) -> str:
        """Show every streamed part in one live panel and return the reply's content text.

        Args:
            streaming_response: The raw chunk stream from the model.
            tool_calls_out: Filled with the assembled tool calls.
            current_loop: The loop number shown in the panel title.

        Returns:
            str: The reply's content text, without reasoning or tool calls.
        """
        content_out: list = []
        styles = {"reasoning": "italic grey50", "tool_call": "cyan"}

        formatter.print_streaming_panel(
            (
                (text, styles.get(kind))
                for kind, text, _ in self.stream_parts(
                    streaming_response, tool_calls_out, content_out
                )
            ),
            title=f"🤖 Agent: {self.agent.agent_name} Loops: {current_loop}",
            style=None,  # Use random color like non-streaming approach
            collect_chunks=True,
        )
        return "".join(content_out)

    # Async invocation.

    async def _arun_llm(self, llm: Any, **run_args) -> Any:
        """Await one model call, on a thread only when the model has no arun.

        Args:
            llm: The model client to call.
            **run_args: Passed through to the model.

        Returns:
            Any: The model's reply, or an async chunk stream.
        """
        if hasattr(llm, "arun"):
            return await llm.arun(**run_args)
        return await asyncio.to_thread(llm.run, **run_args)

    async def _acall_llm(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        current_loop: int = 0,
        streaming_callback: Optional[Callable[[str], Any]] = None,
        *args,
        **kwargs,
    ) -> Any:
        """Await the model call for one loop, picking the streaming mode from the agent's config.

        Args:
            task (str): The task or prompt to send to the LLM.
            img (Optional[str]): Image input for multimodal processing.
            imgs (Optional[List[str]]): Several image inputs.
            current_loop (int): Loop iteration, used in streaming panel titles.
            streaming_callback (Optional[Callable[[str], Any]]): Receives streamed tokens.
            *args: Passed through to the LLM.
            **kwargs: Passed through to the LLM.

        Returns:
            Any: The complete response string, or the tool-call list.
        """
        agent = self.agent
        kwargs.pop("is_last", None)
        if imgs:
            kwargs["imgs"] = imgs
        if img is not None:
            kwargs["img"] = img

        streaming = (
            agent.stream
            or agent.streaming_on
            or streaming_callback is not None
        ) and hasattr(agent.llm, "stream")
        if not streaming:
            return await self._arun_llm(
                agent.llm, task=task, **kwargs
            )
        return await self._astream_call(
            agent.llm,
            current_loop,
            streaming_callback,
            task,
            *args,
            **kwargs,
        )

    async def _astream_call(
        self,
        llm: Any,
        current_loop: int,
        streaming_callback: Optional[Callable[[str], Any]],
        task: str,
        *args,
        **kwargs,
    ) -> Any:
        """Await a streamed model call and consume it in the agent's streaming mode.

        Args:
            llm: The model client to call.
            current_loop (int): The loop number shown in the panel title.
            streaming_callback: Receives streamed tokens, if given.
            task (str): The prompt to send.
            *args: Passed through to the LLM.
            **kwargs: Passed through to the LLM.

        Returns:
            Any: The assembled tool calls if the model made any, else the reply text.
        """
        original_stream = llm.stream
        llm.stream = True
        try:
            if hasattr(llm, "arun"):
                response = await llm.arun(task, *args, **kwargs)
            else:
                response = await asyncio.to_thread(
                    llm.run, task, *args, **kwargs
                )
            # A client without arun hands back a sync stream, read off the loop.
            if hasattr(response, "__iter__") and not isinstance(
                response, (str, list, dict)
            ):
                response = _aiter_sync(response)
            if not hasattr(response, "__aiter__"):
                return response
            if self.agent.stream:
                return await self._aconsume_detailed(
                    response, streaming_callback
                )
            return await self._aconsume_panel(
                response, current_loop, streaming_callback
            )
        finally:
            llm.stream = original_stream

    async def _aconsume_detailed(
        self,
        streaming_response,
        streaming_callback: Optional[Callable[[dict], Any]],
    ) -> Any:
        """Print and forward a token_info dict for every part of an async stream.

        Args:
            streaming_response: The raw async chunk stream from the model.
            streaming_callback: Receives each token_info dict, if given.

        Returns:
            Any: The assembled tool calls if the model made any, else the reply text.
        """
        token_count = 0
        last_chunk = [None]
        tool_calls_out: list = []
        content_out: list = []

        async def track_last(stream):
            async for chunk in stream:
                last_chunk[0] = chunk
                yield chunk

        async for kind, text, chunk in self.astream_parts(
            track_last(streaming_response),
            tool_calls_out,
            content_out,
        ):
            token_count += 1
            token_info = self._build_token_info(
                chunk or last_chunk[0], text, token_count, kind
            )
            print(f"ResponseStream {token_info}")
            await emit_token(streaming_callback, token_info)

        print(self._format_final_chunk(last_chunk[0]))
        return (
            tool_calls_out if tool_calls_out else "".join(content_out)
        )

    async def _aconsume_panel(
        self,
        streaming_response,
        current_loop: int,
        streaming_callback: Optional[Callable[[str], Any]],
    ) -> Any:
        """Send every part of an async stream to the callback, the live panel, or nowhere.

        Args:
            streaming_response: The raw async chunk stream from the model.
            current_loop: The loop number shown in the panel title.
            streaming_callback: Receives the text of every part, if given.

        Returns:
            Any: The assembled tool calls if the model made any, else the reply text.
        """
        tool_calls_out: list = []
        content_out: list = []
        parts = self.astream_parts(
            streaming_response, tool_calls_out, content_out
        )

        # A callback takes precedence over the panel, as in the sync path.
        if (
            streaming_callback is not None
            or self.agent.print_on is False
        ):
            async for _, text, _ in parts:
                await emit_token(streaming_callback, text)
        else:
            styles = {
                "reasoning": "italic grey50",
                "tool_call": "cyan",
            }
            with formatter.streaming_panel(
                f"🤖 Agent: {self.agent.agent_name} Loops: {current_loop}"
            ) as add:
                async for kind, text, _ in parts:
                    add(text, styles.get(kind))

        return (
            tool_calls_out if tool_calls_out else "".join(content_out)
        )

    async def arun_side_call(
        self,
        llm: Any,
        task: str,
        current_loop: int = 0,
        streaming_callback: Optional[Callable[[str], Any]] = None,
    ) -> Any:
        """Await a call on a model other than the agent's own, streamed the same way when streaming is on.

        Args:
            llm: The model to call, such as the tool summary model.
            task: The prompt to send.
            current_loop: The loop number shown in the panel title.
            streaming_callback: Receives streamed tokens. Defaults to the callback of the run in progress.

        Returns:
            Any: The model's reply.
        """
        agent = self.agent
        if streaming_callback is None:
            streaming_callback = getattr(
                agent, "_run_streaming_callback", None
            ) or getattr(agent, "streaming_callback", None)

        if not (
            agent.stream
            or agent.streaming_on
            or streaming_callback is not None
        ) or not hasattr(llm, "stream"):
            return await self._arun_llm(llm, task=task)
        return await self._astream_call(
            llm, current_loop, streaming_callback, task
        )

    def run_side_call(
        self,
        llm: Any,
        task: str,
        current_loop: int = 0,
        streaming_callback: Optional[Callable[[str], None]] = None,
    ) -> Any:
        """Run a call on a model other than the agent's own, streaming it the same way when streaming is on.

        Args:
            llm: The model to call, such as the tool summary model.
            task: The prompt to send.
            current_loop: The loop number shown in the panel title.
            streaming_callback: Receives streamed tokens. Defaults to the callback of the run in progress.

        Returns:
            Any: The model's reply.
        """
        agent = self.agent
        if streaming_callback is None:
            streaming_callback = getattr(
                agent, "_run_streaming_callback", None
            ) or getattr(agent, "streaming_callback", None)

        if not (
            agent.stream
            or agent.streaming_on
            or streaming_callback is not None
        ) or not hasattr(llm, "stream"):
            return llm.run(task)

        llm.stream = True
        response = llm.run(task)
        if not hasattr(response, "__iter__") or isinstance(
            response, str
        ):
            return response
        if agent.stream:
            return self._consume_detailed(
                response, streaming_callback
            )
        return self._consume_panel(
            response, current_loop, streaming_callback
        )
