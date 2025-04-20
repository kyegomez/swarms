from __future__ import annotations

import abc
import asyncio
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal
from typing_extensions import NotRequired, TypedDict

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from loguru import logger
from mcp import ClientSession, StdioServerParameters, Tool as MCPTool, stdio_client
from mcp.client.sse import sse_client
from mcp.types import CallToolResult, JSONRPCMessage

from swarms.utils.any_to_str import any_to_str


class MCPServer(abc.ABC):
    """Base class for Model Context Protocol servers."""

    @abc.abstractmethod
    async def connect(self) -> None:
        """Establish connection to the MCP server."""
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable server name."""
        pass

    @abc.abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources and close connection."""
        pass

    @abc.abstractmethod
    async def list_tools(self) -> List[MCPTool]:
        """List available MCP tools on the server."""
        pass

    @abc.abstractmethod
    async def call_tool(
        self, tool_name: str, arguments: Dict[str, Any] | None
    ) -> CallToolResult:
        """Invoke a tool by name with provided arguments."""
        pass


class _MCPServerWithClientSession(MCPServer, abc.ABC):
    """Mixin providing ClientSession-based MCP communication."""

    def __init__(self, cache_tools_list: bool = False):
        self.session: Optional[ClientSession] = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self._cleanup_lock = asyncio.Lock()
        self.cache_tools_list = cache_tools_list
        self._cache_dirty = True
        self._tools_list: Optional[List[MCPTool]] = None

    @abc.abstractmethod
    def create_streams(
        self
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[JSONRPCMessage | Exception],
            MemoryObjectSendStream[JSONRPCMessage],
        ]
    ]:
        """Supply the read/write streams for the MCP transport."""
        pass

    async def __aenter__(self) -> MCPServer:
        await self.connect()
        return self  # type: ignore

    async def __aexit__(self, exc_type, exc_value, tb) -> None:
        await self.cleanup()

    async def connect(self) -> None:
        """Initialize transport and ClientSession."""
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

    async def cleanup(self) -> None:
        """Close session and transport."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
            finally:
                self.session = None

    async def list_tools(self) -> List[MCPTool]:
        if not self.session:
            raise RuntimeError("Server not connected. Call connect() first.")
        if self.cache_tools_list and not self._cache_dirty and self._tools_list:
            return self._tools_list
        self._cache_dirty = False
        self._tools_list = (await self.session.list_tools()).tools
        return self._tools_list  # type: ignore

    async def call_tool(
        self, tool_name: str | None = None, arguments: Dict[str, Any] | None = None
    ) -> CallToolResult:
        if not arguments:
            raise ValueError("Arguments dict is required to call a tool")
        name = tool_name or arguments.get("tool_name") or arguments.get("name")
        if not name:
            raise ValueError("Tool name missing in arguments")
        if not self.session:
            raise RuntimeError("Server not connected. Call connect() first.")
        return await self.session.call_tool(name, arguments)


class MCPServerStdioParams(TypedDict):
    """Configuration for stdio transport."""
    command: str
    args: NotRequired[List[str]]
    env: NotRequired[Dict[str, str]]
    cwd: NotRequired[str | Path]
    encoding: NotRequired[str]
    encoding_error_handler: NotRequired[Literal["strict", "ignore", "replace"]]


class MCPServerStdio(_MCPServerWithClientSession):
    """MCP server over stdio transport."""

    def __init__(
        self,
        params: MCPServerStdioParams,
        cache_tools_list: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(cache_tools_list)
        self.params = StdioServerParameters(
            command=params["command"],
            args=params.get("args", []),
            env=params.get("env"),
            cwd=params.get("cwd"),
            encoding=params.get("encoding", "utf-8"),
            encoding_error_handler=params.get("encoding_error_handler", "strict"),
        )
        self._name = name or f"stdio:{self.params.command}"

    def create_streams(self) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[JSONRPCMessage | Exception],
            MemoryObjectSendStream[JSONRPCMessage],
        ]
    ]:
        return stdio_client(self.params)

    @property
    def name(self) -> str:
        return self._name


class MCPServerSseParams(TypedDict):
    """Configuration for HTTP+SSE transport."""
    url: str
    headers: NotRequired[Dict[str, str]]
    timeout: NotRequired[float]
    sse_read_timeout: NotRequired[float]


class MCPServerSse(_MCPServerWithClientSession):
    """MCP server over HTTP with SSE transport."""

    def __init__(
        self,
        params: MCPServerSseParams,
        cache_tools_list: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(cache_tools_list)
        self.params = params
        self._name = name or f"sse:{params['url']}"

    def create_streams(self) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[JSONRPCMessage | Exception],
            MemoryObjectSendStream[JSONRPCMessage],
        ]
    ]:
        return sse_client(
            url=self.params["url"],
            headers=self.params.get("headers"),
            timeout=self.params.get("timeout", 5),
            sse_read_timeout=self.params.get("sse_read_timeout", 300),
        )

    @property
    def name(self) -> str:
        return self._name


async def call_tool_fast(
    server: MCPServerSse, payload: Dict[str, Any] | str
) -> Any:
    try:
        await server.connect()
        result = await server.call_tool(arguments=payload if isinstance(payload, dict) else None)
        return result
    finally:
        await server.cleanup()


async def mcp_flow_get_tool_schema(
    params: MCPServerSseParams,
) -> Any:
    async with MCPServerSse(params) as server:
        tools = await server.list_tools()
        return tools


async def mcp_flow(
    params: MCPServerSseParams,
    function_call: Dict[str, Any] | str,
) -> Any:
    async with MCPServerSse(params) as server:
        return await call_tool_fast(server, function_call)


async def _call_one_server(
    params: MCPServerSseParams, payload: Dict[str, Any] | str
) -> Any:
    server = MCPServerSse(params)
    try:
        await server.connect()
        return await server.call_tool(arguments=payload if isinstance(payload, dict) else None)
    finally:
        await server.cleanup()


def batch_mcp_flow(
    params: List[MCPServerSseParams], payload: Dict[str, Any] | str
) -> List[Any]:
    return asyncio.run(
        asyncio.gather(*[_call_one_server(p, payload) for p in params])
    )