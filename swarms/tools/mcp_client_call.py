import os
import asyncio
import contextlib
import json
import random
from functools import wraps
from typing import Any, Dict, List, Literal, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from litellm.types.utils import ChatCompletionMessageToolCall
from loguru import logger
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import (
    CallToolRequestParams as MCPCallToolRequestParams,
)
from mcp.types import CallToolResult as MCPCallToolResult
from mcp.types import Tool as MCPTool
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params.function_definition import (
    FunctionDefinition,
)

from swarms.schemas.mcp_schemas import (
    MCPConnection,
)
from swarms.utils.index import exists


class MCPError(Exception):
    """Base exception for MCP related errors."""

    pass


class MCPConnectionError(MCPError):
    """Raised when there are issues connecting to the MCP server."""

    pass


class MCPToolError(MCPError):
    """Raised when there are issues with MCP tool operations."""

    pass


class MCPValidationError(MCPError):
    """Raised when there are validation issues with MCP operations."""

    pass


class MCPExecutionError(MCPError):
    """Raised when there are issues executing MCP operations."""

    pass


########################################################
# List MCP Tool functions
########################################################
def transform_mcp_tool_to_openai_tool(
    mcp_tool: MCPTool,
) -> ChatCompletionToolParam:
    """Convert an MCP tool to an OpenAI tool."""
    return ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name=mcp_tool.name,
            description=mcp_tool.description or "",
            parameters=mcp_tool.inputSchema,
            strict=False,
        ),
    )


async def load_mcp_tools(
    session: ClientSession, format: Literal["mcp", "openai"] = "mcp"
) -> Union[List[MCPTool], List[ChatCompletionToolParam]]:
    """
    Load all available MCP tools

    Args:
        session: The MCP session to use
        format: The format to convert the tools to
    By default, the tools are returned in MCP format.

    If format is set to "openai", the tools are converted to OpenAI API compatible tools.
    """
    tools = await session.list_tools()
    if format == "openai":
        return [
            transform_mcp_tool_to_openai_tool(mcp_tool=tool)
            for tool in tools.tools
        ]
    return tools.tools


########################################################
# Call MCP Tool functions
########################################################


async def call_mcp_tool(
    session: ClientSession,
    call_tool_request_params: MCPCallToolRequestParams,
) -> MCPCallToolResult:
    """Call an MCP tool."""
    tool_result = await session.call_tool(
        name=call_tool_request_params.name,
        arguments=call_tool_request_params.arguments,
    )
    return tool_result


def _get_function_arguments(function: FunctionDefinition) -> dict:
    """Helper to safely get and parse function arguments."""
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    return arguments if isinstance(arguments, dict) else {}


def transform_openai_tool_call_request_to_mcp_tool_call_request(
    openai_tool: Union[ChatCompletionMessageToolCall, Dict],
) -> MCPCallToolRequestParams:
    """Convert an OpenAI ChatCompletionMessageToolCall to an MCP CallToolRequestParams."""
    function = openai_tool["function"]
    return MCPCallToolRequestParams(
        name=function["name"],
        arguments=_get_function_arguments(function),
    )


async def call_openai_tool(
    session: ClientSession,
    openai_tool: dict,
) -> MCPCallToolResult:
    """
    Call an OpenAI tool using MCP client.

    Args:
        session: The MCP session to use
        openai_tool: The OpenAI tool to call. You can get this from the `choices[0].message.tool_calls[0]` of the response from the OpenAI API.
    Returns:
        The result of the MCP tool call.
    """
    mcp_tool_call_request_params = (
        transform_openai_tool_call_request_to_mcp_tool_call_request(
            openai_tool=openai_tool,
        )
    )
    return await call_mcp_tool(
        session=session,
        call_tool_request_params=mcp_tool_call_request_params,
    )


def retry_with_backoff(retries=3, backoff_in_seconds=1):
    """Decorator for retrying functions with exponential backoff."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        logger.error(
                            f"Failed after {retries} retries: {str(e)}"
                        )
                        raise
                    sleep_time = (
                        backoff_in_seconds * 2**x
                        + random.uniform(0, 1)
                    )
                    logger.warning(
                        f"Attempt {x + 1} failed, retrying in {sleep_time:.2f}s"
                    )
                    await asyncio.sleep(sleep_time)
                    x += 1

        return wrapper

    return decorator


@contextlib.contextmanager
def get_or_create_event_loop():
    """Context manager to handle event loop creation and cleanup."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        yield loop
    finally:
        # Only close the loop if we created it and it's not the main event loop
        if loop != asyncio.get_event_loop() and not loop.is_running():
            if not loop.is_closed():
                loop.close()


def connect_to_mcp_server(connection: MCPConnection = None):
    """Connect to an MCP server.

    Args:
        connection (MCPConnection): The connection configuration object

    Returns:
        tuple: A tuple containing (headers, timeout, transport, url)

    Raises:
        MCPValidationError: If the connection object is invalid
    """
    if not isinstance(connection, MCPConnection):
        raise MCPValidationError("Invalid connection type")

    # Direct attribute access is faster than property access
    headers = dict(connection.headers or {})
    if connection.authorization_token:
        headers["Authorization"] = (
            f"Bearer {connection.authorization_token}"
        )

    return (
        headers,
        connection.timeout or 5,
        connection.transport or "sse",
        connection.url,
    )


@retry_with_backoff(retries=3)
async def aget_mcp_tools(
    server_path: Optional[str] = None,
    format: str = "openai",
    connection: Optional[MCPConnection] = None,
    *args,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Fetch available MCP tools from the server with retry logic.

    Args:
        server_path (str): Path to the MCP server script

    Returns:
        List[Dict[str, Any]]: List of available MCP tools in OpenAI format

    Raises:
        MCPValidationError: If server_path is invalid
        MCPConnectionError: If connection to server fails
    """
    if exists(connection):
        headers, timeout, transport, url = connect_to_mcp_server(
            connection
        )
    else:
        headers, timeout, _transport, _url = (
            None,
            5,
            None,
            server_path,
        )

    logger.info(f"Fetching MCP tools from server: {server_path}")

    try:
        async with sse_client(
            url=server_path,
            headers=headers,
            timeout=timeout,
            *args,
            **kwargs,
        ) as (
            read,
            write,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(
                    session=session, format=format
                )
                logger.info(
                    f"Successfully fetched {len(tools)} tools"
                )
                return tools
    except Exception as e:
        logger.error(f"Error fetching MCP tools: {str(e)}")
        raise MCPConnectionError(
            f"Failed to connect to MCP server: {str(e)}"
        )


def get_mcp_tools_sync(
    server_path: Optional[str] = None,
    format: str = "openai",
    connection: Optional[MCPConnection] = None,
    *args,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Synchronous version of get_mcp_tools that handles event loop management.

    Args:
        server_path (str): Path to the MCP server script

    Returns:
        List[Dict[str, Any]]: List of available MCP tools in OpenAI format

    Raises:
        MCPValidationError: If server_path is invalid
        MCPConnectionError: If connection to server fails
        MCPExecutionError: If event loop management fails
    """
    with get_or_create_event_loop() as loop:
        try:
            return loop.run_until_complete(
                aget_mcp_tools(
                    server_path=server_path,
                    format=format,
                    connection=connection,
                    *args,
                    **kwargs,
                )
            )
        except Exception as e:
            logger.error(f"Error in get_mcp_tools_sync: {str(e)}")
            raise MCPExecutionError(
                f"Failed to execute MCP tools sync: {str(e)}"
            )


def _fetch_tools_for_server(
    url: str,
    connection: Optional[MCPConnection] = None,
    format: str = "openai",
) -> List[Dict[str, Any]]:
    """Helper function to fetch tools for a single server."""
    return get_mcp_tools_sync(
        server_path=url,
        connection=connection,
        format=format,
    )


def get_tools_for_multiple_mcp_servers(
    urls: List[str],
    connections: List[MCPConnection] = None,
    format: str = "openai",
    output_type: Literal["json", "dict", "str"] = "str",
    max_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Get tools for multiple MCP servers concurrently using ThreadPoolExecutor.

    Args:
        urls: List of server URLs to fetch tools from
        connections: Optional list of MCPConnection objects corresponding to each URL
        format: Format to return tools in (default: "openai")
        output_type: Type of output format (default: "str")
        max_workers: Maximum number of worker threads (default: None, uses min(32, os.cpu_count() + 4))

    Returns:
        List[Dict[str, Any]]: Combined list of tools from all servers
    """
    tools = []
    (
        min(32, os.cpu_count() + 4)
        if max_workers is None
        else max_workers
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if exists(connections):
            # Create future tasks for each URL-connection pair
            future_to_url = {
                executor.submit(
                    _fetch_tools_for_server, url, connection, format
                ): url
                for url, connection in zip(urls, connections)
            }
        else:
            # Create future tasks for each URL without connections
            future_to_url = {
                executor.submit(
                    _fetch_tools_for_server, url, None, format
                ): url
                for url in urls
            }

        # Process completed futures as they come in
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                server_tools = future.result()
                tools.extend(server_tools)
            except Exception as e:
                logger.error(
                    f"Error fetching tools from {url}: {str(e)}"
                )
                raise MCPExecutionError(
                    f"Failed to fetch tools from {url}: {str(e)}"
                )

    return tools


async def _execute_tool_call_simple(
    response: any = None,
    server_path: str = None,
    connection: Optional[MCPConnection] = None,
    output_type: Literal["json", "dict", "str"] = "str",
    *args,
    **kwargs,
):
    """Execute a tool call using the MCP client."""
    if exists(connection):
        headers, timeout, transport, url = connect_to_mcp_server(
            connection
        )
    else:
        headers, timeout, _transport, url = (
            None,
            5,
            "sse",
            server_path,
        )

    try:
        async with sse_client(
            url=url, headers=headers, timeout=timeout, *args, **kwargs
        ) as (
            read,
            write,
        ):
            async with ClientSession(read, write) as session:
                try:
                    await session.initialize()

                    call_result = await call_openai_tool(
                        session=session,
                        openai_tool=response,
                    )

                    if output_type == "json":
                        out = call_result.model_dump_json(indent=4)
                    elif output_type == "dict":
                        out = call_result.model_dump()
                    elif output_type == "str":
                        data = call_result.model_dump()
                        formatted_lines = []
                        for key, value in data.items():
                            if isinstance(value, list):
                                for item in value:
                                    if isinstance(item, dict):
                                        for k, v in item.items():
                                            formatted_lines.append(
                                                f"{k}: {v}"
                                            )
                            else:
                                formatted_lines.append(
                                    f"{key}: {value}"
                                )
                        out = "\n".join(formatted_lines)

                    return out

                except Exception as e:
                    logger.error(f"Error in tool execution: {str(e)}")
                    raise MCPExecutionError(
                        f"Tool execution failed: {str(e)}"
                    )

    except Exception as e:
        logger.error(f"Error in SSE client connection: {str(e)}")
        raise MCPConnectionError(
            f"Failed to connect to MCP server: {str(e)}"
        )


async def execute_tool_call_simple(
    response: any = None,
    server_path: str = None,
    connection: Optional[MCPConnection] = None,
    output_type: Literal["json", "dict", "str", "formatted"] = "str",
    *args,
    **kwargs,
) -> List[Dict[str, Any]]:
    return await _execute_tool_call_simple(
        response=response,
        server_path=server_path,
        connection=connection,
        output_type=output_type,
        *args,
        **kwargs,
    )


async def execute_mcp_call(
    function_name: str,
    server_url: str,
    payload: Dict[str, Any],
    connection: Optional[MCPConnection] = None,
    output_type: Literal["json", "dict", "str"] = "str",
    *args,
    **kwargs,
) -> Any:
    """Execute a specific MCP tool call on a server.

    Parameters
    ----------
    function_name: str
        Name of the MCP tool to execute.
    server_url: str
        URL of the MCP server.
    payload: Dict[str, Any]
        Arguments to pass to the MCP tool.
    connection: Optional[MCPConnection]
        Optional connection configuration.
    output_type: str
        Output formatting type.
    """

    if exists(connection):
        headers, timeout, _transport, url = connect_to_mcp_server(
            connection
        )
    else:
        headers, timeout, _transport, url = None, 5, None, server_url

    try:
        async with sse_client(
            url=url, headers=headers, timeout=timeout, *args, **kwargs
        ) as (
            read,
            write,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                req = MCPCallToolRequestParams(
                    name=function_name, arguments=payload
                )
                result = await call_mcp_tool(
                    session=session, call_tool_request_params=req
                )

                if output_type == "json":
                    return result.model_dump_json(indent=4)
                if output_type == "dict":
                    return result.model_dump()

                data = result.model_dump()
                formatted_lines = []
                for key, value in data.items():
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                for k, v in item.items():
                                    formatted_lines.append(
                                        f"{k}: {v}"
                                    )
                    else:
                        formatted_lines.append(f"{key}: {value}")
                return "\n".join(formatted_lines)
    except Exception as e:
        logger.error(f"Error executing MCP call: {e}")
        raise MCPExecutionError(f"Failed to execute MCP call: {e}")
