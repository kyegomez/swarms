from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import NotRequired, TypedDict
from contextlib import AbstractAsyncContextManager
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client, StdioServerParameters

from loguru import logger
import abc
import asyncio
from pathlib import Path
from typing import Literal

from anyio.streams.memory import (
    MemoryObjectReceiveStream,
    MemoryObjectSendStream,
)
from mcp.types import CallToolResult, JSONRPCMessage # Kept for backward compatibility, might be removed later

from swarms.utils.any_to_str import any_to_str
from mcp import (
    ClientSession as OldClientSession, # Kept for backward compatibility with stdio
    StdioServerParameters,
    Tool as MCPTool,
    stdio_client,
)

class MCPServer(abc.ABC):
    """Base class for Model Context Protocol servers."""

    @abc.abstractmethod
    async def connect(self):
        """Connect to the server. For example, this might mean spawning a subprocess or
        opening a network connection. The server is expected to remain connected until
        `cleanup()` is called.
        """
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """A readable name for the server."""
        pass

    @abc.abstractmethod
    async def cleanup(self):
        """Cleanup the server. For example, this might mean closing a subprocess or
        closing a network connection.
        """
        pass

    @abc.abstractmethod
    async def list_tools(self) -> list[Any]: # Changed to Any for flexibility
        """List the tools available on the server."""
        pass

    @abc.abstractmethod
    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None
    ) -> CallToolResult: # Kept for backward compatibility, might be removed later
        """Invoke a tool on the server."""
        pass



class _MCPServerWithClientSession(MCPServer, abc.ABC):
    """Base class for MCP servers that use a `ClientSession` to communicate with the server."""

    def __init__(self, cache_tools_list: bool):
        """
        Args:
            cache_tools_list: Whether to cache the tools list. If `True`, the tools list will be
            cached and only fetched from the server once. If `False`, the tools list will be
            fetched from the server on each call to `list_tools()`. The cache can be invalidated
            by calling `invalidate_tools_cache()`. You should set this to `True` if you know the
            server will not change its tools list, because it can drastically improve latency
            (by avoiding a round-trip to the server every time).
        """
        self.session: ClientSession | None = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.cache_tools_list = cache_tools_list

        # The cache is always dirty at startup, so that we fetch tools at least once
        self._cache_dirty = True
        self._tools_list: list[Any] | None = None # Changed to Any for flexibility

    @abc.abstractmethod
    def create_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[JSONRPCMessage | Exception],
            MemoryObjectSendStream[JSONRPCMessage],
        ]
    ]:
        """Create the streams for the server."""
        pass

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.cleanup()

    def invalidate_tools_cache(self):
        """Invalidate the tools cache."""
        self._cache_dirty = True

    async def connect(self):
        """Connect to the server."""
        try:
            transport = await self.exit_stack.enter_async_context(
                self.create_streams()
            )
            read, write = transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
        except Exception as e:
            logger.error(f"Error initializing MCP server: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> list[Any]: # Changed to Any for flexibility
        """List the tools available on the server."""
        if not self.session:
            raise Exception(
                "Server not initialized. Make sure you call `connect()` first."
            )

        # Return from cache if caching is enabled, we have tools, and the cache is not dirty
        if (
            self.cache_tools_list
            and not self._cache_dirty
            and self._tools_list
        ):
            return self._tools_list

        # Reset the cache dirty to False
        self._cache_dirty = False

        # Fetch the tools from the server
        self._tools_list = (await self.session.list_tools()).tools
        return self._tools_list

    async def call_tool(
        self, arguments: dict[str, Any] | None
    ) -> CallToolResult: # Kept for backward compatibility, might be removed later
        """Invoke a tool on the server."""
        tool_name = arguments.get("tool_name") or arguments.get(
            "name"
        )

        if not tool_name:
            raise Exception("No tool name found in arguments")

        if not self.session:
            raise Exception(
                "Server not initialized. Make sure you call `connect()` first."
            )

        return await self.session.call_tool(tool_name, arguments)

    async def cleanup(self):
        """Cleanup the server."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
            except Exception as e:
                logger.error(f"Error cleaning up server: {e}")


class MCPServerStdioParams(TypedDict):
    """Mirrors `mcp.client.stdio.StdioServerParameters`, but lets you pass params without another
    import.
    """

    command: str
    """The executable to run to start the server. For example, `python` or `node`."""

    args: NotRequired[list[str]]
    """Command line args to pass to the `command` executable. For example, `['foo.py']` or
    `['server.js', '--port', '8080']`."""

    env: NotRequired[dict[str, str]]
    """The environment variables to set for the server. ."""

    cwd: NotRequired[str | Path]
    """The working directory to use when spawning the process."""

    encoding: NotRequired[str]
    """The text encoding used when sending/receiving messages to the server. Defaults to `utf-8`."""

    encoding_error_handler: NotRequired[
        Literal["strict", "ignore", "replace"]
    ]
    """The text encoding error handler. Defaults to `strict`.

    See https://docs.python.org/3/library/codecs.html#codec-base-classes for
    explanations of possible values.
    """


class MCPServerStdio(_MCPServerWithClientSession):
    """MCP server implementation that uses the stdio transport. See the [spec]
    (https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#stdio) for
    details.
    """

    def __init__(
        self,
        params: MCPServerStdioParams,
        cache_tools_list: bool = False,
        name: str | None = None,
    ):
        """Create a new MCP server based on the stdio transport.

        Args:
            params: The params that configure the server. This includes the command to run to
                start the server, the args to pass to the command, the environment variables to
                set for the server, the working directory to use when spawning the process, and
                the text encoding used when sending/receiving messages to the server.
            cache_tools_list: Whether to cache the tools list. If `True`, the tools list will be
                cached and only fetched from the server once. If `False`, the tools list will be
                fetched from the server on each call to `list_tools()`. The cache can be
                invalidated by calling `invalidate_tools_cache()`. You should set this to `True`
                if you know the server will not change its tools list, because it can drastically
                improve latency (by avoiding a round-trip to the server every time).
            name: A readable name for the server. If not provided, we'll create one from the
                command.
        """
        super().__init__(cache_tools_list)

        self.params = StdioServerParameters(
            command=params["command"],
            args=params.get("args", []),
            env=params.get("env"),
            cwd=params.get("cwd"),
            encoding=params.get("encoding", "utf-8"),
            encoding_error_handler=params.get(
                "encoding_error_handler", "strict"
            ),
        )

        self._name = name or f"stdio: {self.params.command}"

    def create_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[JSONRPCMessage | Exception],
            MemoryObjectSendStream[JSONRPCMessage],
        ]
    ]:
        """Create the streams for the server."""
        return stdio_client(self.params)

    @property
    def name(self) -> str:
        """A readable name for the server."""
        return self._name



class MCPServerSseParams(TypedDict):
    """Mirrors the params in`mcp.client.sse.sse_client`."""

    url: str
    """The URL of the server."""

    headers: NotRequired[dict[str, str]]
    """The headers to send to the server."""

    timeout: NotRequired[float]
    """The timeout for the HTTP request. Defaults to 5 seconds."""

    sse_read_timeout: NotRequired[float]
    """The timeout for the SSE connection, in seconds. Defaults to 5 minutes."""


class MCPServerSse:
    def __init__(self, params: MCPServerSseParams):
        self.params = params
        self.client: Optional[ClientSession] = None
        self._connection_lock = asyncio.Lock()
        self.messages = []  # Store messages instead of using conversation
        self.preserve_format = True  # Flag to preserve original formatting

    async def connect(self):
        """Connect to the MCP server with proper locking."""
        async with self._connection_lock:
            if not self.client:
                transport = await self.create_streams()
                read_stream, write_stream = transport
                self.client = ClientSession(read_stream=read_stream, write_stream=write_stream)
                await self.client.initialize()

    def create_streams(self, **kwargs) -> AbstractAsyncContextManager[Any]:
        return sse_client(
            url=self.params["url"],
            headers=self.params.get("headers", None),
            timeout=self.params.get("timeout", 5),
            sse_read_timeout=self.params.get("sse_read_timeout", 60 * 5),
        )

    def _parse_input(self, payload: Any) -> dict:
        """Parse input while preserving original format."""
        if isinstance(payload, dict):
            return payload

        if isinstance(payload, str):
            try:
                # Try to parse as JSON
                import json
                return json.loads(payload)
            except json.JSONDecodeError:
                # Check if it's a math operation
                import re

                # Pattern matching for basic math operations
                add_pattern = r"(?i)(?:what\s+is\s+)?(\d+)\s*(?:plus|\+)\s*(\d+)"
                mult_pattern = r"(?i)(?:multiply|times|\*)\s*(\d+)\s*(?:and|by)?\s*(\d+)"
                div_pattern = r"(?i)(?:divide)\s*(\d+)\s*(?:by)\s*(\d+)"

                # Check for addition
                if match := re.search(add_pattern, payload):
                    a, b = map(int, match.groups())
                    return {"tool_name": "add", "a": a, "b": b}

                # Check for multiplication
                if match := re.search(mult_pattern, payload):
                    a, b = map(int, match.groups())
                    return {"tool_name": "multiply", "a": a, "b": b}

                # Check for division
                if match := re.search(div_pattern, payload):
                    a, b = map(int, match.groups())
                    return {"tool_name": "divide", "a": a, "b": b}

                # Default to text input if no pattern matches
                return {"text": payload}

        return {"text": str(payload)}

    def _format_output(self, result: Any, original_input: Any) -> str:
        """Format output based on input type and result."""
        if not self.preserve_format:
            return str(result)

        try:
            if isinstance(result, (int, float)):
                # For numeric results, format based on operation
                if isinstance(original_input, dict):
                    tool_name = original_input.get("tool_name", "")
                    if tool_name == "add":
                        return f"{original_input['a']} + {original_input['b']} = {result}"
                    elif tool_name == "multiply":
                        return f"{original_input['a']} * {original_input['b']} = {result}"
                    elif tool_name == "divide":
                        return f"{original_input['a']} / {original_input['b']} = {result}"
                return str(result)
            elif isinstance(result, dict):
                return json.dumps(result, indent=2)
            else:
                return str(result)
        except Exception as e:
            logger.error(f"Error formatting output: {e}")
            return str(result)

    async def call_tool(self, payload: Any) -> Any:
        """Call a tool on the MCP server with support for various input formats."""
        if not self.client:
            raise RuntimeError("Not connected to MCP server")

        # Store original input for formatting
        original_input = payload

        # Parse input
        parsed_payload = self._parse_input(payload)

        # Add message to history
        self.messages.append({
            "role": "user",
            "content": str(payload),
            "parsed": parsed_payload
        })

        try:
            result = await self.client.call_tool(parsed_payload)
            formatted_result = self._format_output(result, original_input)

            self.messages.append({
                "role": "assistant",
                "content": formatted_result,
                "raw_result": result
            })

            return formatted_result
        except Exception as e:
            error_msg = f"Error calling tool: {str(e)}"
            self.messages.append({
                "role": "error",
                "content": error_msg,
                "original_input": payload
            })
            raise

    async def cleanup(self):
        """Clean up the connection with proper locking."""
        async with self._connection_lock:
            if self.client:
                await self.client.close()
                self.client = None

    async def list_tools(self) -> list[Any]:
        """List available tools with proper error handling."""
        if not self.client:
            raise RuntimeError("Not connected to MCP server")
        try:
            return await self.client.list_tools()
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return []


async def call_tool_fast(server: MCPServerSse, payload: dict[str, Any] | str):
    """
    Convenience wrapper that opens → calls → closes in one shot with proper error handling.
    """
    try:
        await server.connect()
        result = await server.call_tool(payload)
        return result.model_dump() if hasattr(result, "model_dump") else result
    except Exception as e:
        logger.error(f"Error in call_tool_fast: {e}")
        raise
    finally:
        await server.cleanup()


async def mcp_flow_get_tool_schema(
    params: MCPServerSseParams,
) -> Any:
    """Get tool schema with proper error handling."""
    try:
        async with MCPServerSse(params) as server:
            tools = await server.list_tools()
            return tools.model_dump() if hasattr(tools, "model_dump") else tools
    except Exception as e:
        logger.error(f"Error getting tool schema: {e}")
        raise


async def mcp_flow(
    params: MCPServerSseParams,
    function_call: dict[str, Any] | str,
) -> Any:
    """Execute MCP flow with proper error handling."""
    try:
        async with MCPServerSse(params) as server:
            return await call_tool_fast(server, function_call)
    except Exception as e:
        logger.error(f"Error in MCP flow: {e}")
        raise


async def _call_one_server(param: MCPServerSseParams, payload: dict[str, Any] | str) -> Any:
    """Make a call to a single MCP server with proper async context management."""
    try:
        server = MCPServerSse(param)
        await server.connect()
        result = await server.call_tool(payload)
        return result
    except Exception as e:
        logger.error(f"Error calling server: {e}")
        raise
    finally:
        if 'server' in locals():
            await server.cleanup()


def batch_mcp_flow(params: List[MCPServerSseParams], payload: dict[str, Any] | str) -> List[Any]:
    """Blocking helper that fans out to all MCP servers in params."""
    try:
        return asyncio.run(_batch(params, payload))
    except Exception as e:
        logger.error(f"Error in batch_mcp_flow: {e}")
        return []


async def _batch(params: List[MCPServerSseParams], payload: dict[str, Any] | str) -> List[Any]:
    """Fan out to all MCP servers asynchronously and gather results."""
    try:
        coros = [_call_one_server(p, payload) for p in params]
        results = await asyncio.gather(*coros, return_exceptions=True)
        # Filter out exceptions and convert to strings
        return [any_to_str(r) for r in results if not isinstance(r, Exception)]
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return []