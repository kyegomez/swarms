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
    if isinstance(response, str):
        response = json.loads(response)

    return await _execute_tool_call_simple(
        response=response,
        server_path=server_path,
        connection=connection,
        output_type=output_type,
        *args,
        **kwargs,
    )


def _create_server_tool_mapping(
    urls: List[str],
    connections: List[MCPConnection] = None,
    format: str = "openai",
) -> Dict[str, Dict[str, Any]]:
    """
    Create a mapping of function names to server information for all MCP servers.

    Args:
        urls: List of server URLs
        connections: Optional list of MCPConnection objects
        format: Format to fetch tools in

    Returns:
        Dict mapping function names to server info (url, connection, tool)
    """
    server_tool_mapping = {}

    for i, url in enumerate(urls):
        connection = (
            connections[i]
            if connections and i < len(connections)
            else None
        )

        try:
            # Get tools for this server
            tools = get_mcp_tools_sync(
                server_path=url,
                connection=connection,
                format=format,
            )

            # Create mapping for each tool
            for tool in tools:
                if isinstance(tool, dict) and "function" in tool:
                    function_name = tool["function"]["name"]
                    server_tool_mapping[function_name] = {
                        "url": url,
                        "connection": connection,
                        "tool": tool,
                        "server_index": i,
                    }
                elif hasattr(tool, "name"):
                    # Handle MCPTool objects
                    server_tool_mapping[tool.name] = {
                        "url": url,
                        "connection": connection,
                        "tool": tool,
                        "server_index": i,
                    }

        except Exception as e:
            logger.warning(
                f"Failed to fetch tools from server {url}: {str(e)}"
            )
            continue

    return server_tool_mapping


async def _create_server_tool_mapping_async(
    urls: List[str],
    connections: List[MCPConnection] = None,
    format: str = "openai",
) -> Dict[str, Dict[str, Any]]:
    """
    Async version: Create a mapping of function names to server information for all MCP servers.

    Args:
        urls: List of server URLs
        connections: Optional list of MCPConnection objects
        format: Format to fetch tools in

    Returns:
        Dict mapping function names to server info (url, connection, tool)
    """
    server_tool_mapping = {}

    for i, url in enumerate(urls):
        connection = (
            connections[i]
            if connections and i < len(connections)
            else None
        )

        try:
            # Get tools for this server using async function
            tools = await aget_mcp_tools(
                server_path=url,
                connection=connection,
                format=format,
            )

            # Create mapping for each tool
            for tool in tools:
                if isinstance(tool, dict) and "function" in tool:
                    function_name = tool["function"]["name"]
                    server_tool_mapping[function_name] = {
                        "url": url,
                        "connection": connection,
                        "tool": tool,
                        "server_index": i,
                    }
                elif hasattr(tool, "name"):
                    # Handle MCPTool objects
                    server_tool_mapping[tool.name] = {
                        "url": url,
                        "connection": connection,
                        "tool": tool,
                        "server_index": i,
                    }

        except Exception as e:
            logger.warning(
                f"Failed to fetch tools from server {url}: {str(e)}"
            )
            continue

    return server_tool_mapping


async def _execute_tool_on_server(
    tool_call: Dict[str, Any],
    server_info: Dict[str, Any],
    output_type: Literal["json", "dict", "str", "formatted"] = "str",
) -> Dict[str, Any]:
    """
    Execute a single tool call on a specific server.

    Args:
        tool_call: The tool call to execute
        server_info: Server information from the mapping
        output_type: Output format type

    Returns:
        Execution result with server metadata
    """
    try:
        result = await _execute_tool_call_simple(
            response=tool_call,
            server_path=server_info["url"],
            connection=server_info["connection"],
            output_type=output_type,
        )

        return {
            "server_url": server_info["url"],
            "server_index": server_info["server_index"],
            "function_name": tool_call.get("function", {}).get(
                "name", "unknown"
            ),
            "result": result,
            "status": "success",
        }

    except Exception as e:
        logger.error(
            f"Failed to execute tool on server {server_info['url']}: {str(e)}"
        )
        return {
            "server_url": server_info["url"],
            "server_index": server_info["server_index"],
            "function_name": tool_call.get("function", {}).get(
                "name", "unknown"
            ),
            "result": None,
            "error": str(e),
            "status": "error",
        }


async def execute_multiple_tools_on_multiple_mcp_servers(
    responses: List[Dict[str, Any]],
    urls: List[str],
    connections: List[MCPConnection] = None,
    output_type: Literal["json", "dict", "str", "formatted"] = "str",
    max_concurrent: Optional[int] = None,
    *args,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Execute multiple tool calls across multiple MCP servers.

    This function creates a mapping of function names to servers, then for each response
    that contains tool calls, it finds the appropriate server for each function and
    executes the calls concurrently.

    Args:
        responses: List of responses containing tool calls (OpenAI format)
        urls: List of MCP server URLs
        connections: Optional list of MCPConnection objects corresponding to each URL
        output_type: Output format type for results
        max_concurrent: Maximum number of concurrent executions (default: len(responses))

    Returns:
        List of execution results with server metadata

    Example:
        # Example responses format:
        responses = [
            {
                "function": {
                    "name": "search_web",
                    "arguments": {"query": "python programming"}
                }
            },
            {
                "function": {
                    "name": "search_database",
                    "arguments": {"table": "users", "id": 123}
                }
            }
        ]

        urls = ["http://server1:8000", "http://server2:8000"]

        results = await execute_multiple_tools_on_multiple_mcp_servers(
            responses=responses,
            urls=urls
        )
    """
    if not responses:
        logger.warning("No responses provided for execution")
        return []

    if not urls:
        raise MCPValidationError("No server URLs provided")

    # Create mapping of function names to servers using async version
    logger.info(f"Creating tool mapping for {len(urls)} servers")
    server_tool_mapping = await _create_server_tool_mapping_async(
        urls=urls, connections=connections, format="openai"
    )

    if not server_tool_mapping:
        raise MCPExecutionError(
            "No tools found on any of the provided servers"
        )

    logger.info(
        f"Found {len(server_tool_mapping)} unique functions across all servers"
    )

    # Extract all tool calls from responses
    all_tool_calls = []
    logger.info(
        f"Processing {len(responses)} responses for tool call extraction"
    )

    # Check if responses are individual characters that need to be reconstructed
    if len(responses) > 10 and all(
        isinstance(r, str) and len(r) == 1 for r in responses
    ):
        logger.info(
            "Detected character-by-character response, reconstructing JSON string"
        )
        try:
            reconstructed_response = "".join(responses)
            logger.info(
                f"Reconstructed response length: {len(reconstructed_response)}"
            )
            logger.debug(
                f"Reconstructed response: {reconstructed_response}"
            )

            # Try to parse the reconstructed response to validate it
            try:
                json.loads(reconstructed_response)
                logger.info(
                    "Successfully validated reconstructed JSON response"
                )
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Reconstructed response is not valid JSON: {str(e)}"
                )
                logger.debug(
                    f"First 100 chars: {reconstructed_response[:100]}"
                )
                logger.debug(
                    f"Last 100 chars: {reconstructed_response[-100:]}"
                )

            responses = [reconstructed_response]
        except Exception as e:
            logger.warning(
                f"Failed to reconstruct response from characters: {str(e)}"
            )

    for i, response in enumerate(responses):
        logger.debug(
            f"Processing response {i}: {type(response)} - {response}"
        )

        # Handle JSON string responses
        if isinstance(response, str):
            try:
                response = json.loads(response)
                logger.debug(
                    f"Parsed JSON string response {i}: {response}"
                )
            except json.JSONDecodeError:
                logger.warning(
                    f"Failed to parse JSON response at index {i}: {response}"
                )
                continue

        if isinstance(response, dict):
            # Single tool call
            if "function" in response:
                logger.debug(
                    f"Found single tool call in response {i}: {response['function']}"
                )
                # Parse arguments if they're a JSON string
                if isinstance(
                    response["function"].get("arguments"), str
                ):
                    try:
                        response["function"]["arguments"] = (
                            json.loads(
                                response["function"]["arguments"]
                            )
                        )
                        logger.debug(
                            f"Parsed function arguments: {response['function']['arguments']}"
                        )
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Failed to parse function arguments: {response['function']['arguments']}"
                        )

                all_tool_calls.append((i, response))
            # Multiple tool calls
            elif "tool_calls" in response:
                logger.debug(
                    f"Found multiple tool calls in response {i}: {len(response['tool_calls'])} calls"
                )
                for tool_call in response["tool_calls"]:
                    # Parse arguments if they're a JSON string
                    if isinstance(
                        tool_call.get("function", {}).get(
                            "arguments"
                        ),
                        str,
                    ):
                        try:
                            tool_call["function"]["arguments"] = (
                                json.loads(
                                    tool_call["function"]["arguments"]
                                )
                            )
                            logger.debug(
                                f"Parsed tool call arguments: {tool_call['function']['arguments']}"
                            )
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Failed to parse tool call arguments: {tool_call['function']['arguments']}"
                            )

                    all_tool_calls.append((i, tool_call))
            # Direct tool call
            elif "name" in response and "arguments" in response:
                logger.debug(
                    f"Found direct tool call in response {i}: {response}"
                )
                # Parse arguments if they're a JSON string
                if isinstance(response.get("arguments"), str):
                    try:
                        response["arguments"] = json.loads(
                            response["arguments"]
                        )
                        logger.debug(
                            f"Parsed direct tool call arguments: {response['arguments']}"
                        )
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Failed to parse direct tool call arguments: {response['arguments']}"
                        )

                all_tool_calls.append((i, {"function": response}))
            else:
                logger.debug(
                    f"Response {i} is a dict but doesn't match expected tool call formats: {list(response.keys())}"
                )
        else:
            logger.warning(
                f"Unsupported response type at index {i}: {type(response)}"
            )
            continue

    if not all_tool_calls:
        logger.warning("No tool calls found in responses")
        return []

    logger.info(f"Found {len(all_tool_calls)} tool calls to execute")

    # Execute tool calls concurrently
    max_concurrent = max_concurrent or len(all_tool_calls)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_with_semaphore(tool_call_info):
        async with semaphore:
            response_index, tool_call = tool_call_info
            function_name = tool_call.get("function", {}).get(
                "name", "unknown"
            )

            if function_name not in server_tool_mapping:
                logger.warning(
                    f"Function '{function_name}' not found on any server"
                )
                return {
                    "response_index": response_index,
                    "function_name": function_name,
                    "result": None,
                    "error": f"Function '{function_name}' not available on any server",
                    "status": "not_found",
                }

            server_info = server_tool_mapping[function_name]
            result = await _execute_tool_on_server(
                tool_call=tool_call,
                server_info=server_info,
                output_type=output_type,
            )
            result["response_index"] = response_index
            return result

    # Execute all tool calls concurrently
    tasks = [
        execute_with_semaphore(tool_call_info)
        for tool_call_info in all_tool_calls
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results and handle exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(
                f"Task {i} failed with exception: {str(result)}"
            )
            processed_results.append(
                {
                    "response_index": (
                        all_tool_calls[i][0]
                        if i < len(all_tool_calls)
                        else -1
                    ),
                    "function_name": "unknown",
                    "result": None,
                    "error": str(result),
                    "status": "exception",
                }
            )
        else:
            processed_results.append(result)

    logger.info(
        f"Completed execution of {len(processed_results)} tool calls"
    )
    return processed_results


def execute_multiple_tools_on_multiple_mcp_servers_sync(
    responses: List[Dict[str, Any]],
    urls: List[str],
    connections: List[MCPConnection] = None,
    output_type: Literal["json", "dict", "str", "formatted"] = "str",
    max_concurrent: Optional[int] = None,
    *args,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Synchronous version of execute_multiple_tools_on_multiple_mcp_servers.

    Args:
        responses: List of responses containing tool calls (OpenAI format)
        urls: List of MCP server URLs
        connections: Optional list of MCPConnection objects corresponding to each URL
        output_type: Output format type for results
        max_concurrent: Maximum number of concurrent executions

    Returns:
        List of execution results with server metadata
    """
    with get_or_create_event_loop() as loop:
        try:
            return loop.run_until_complete(
                execute_multiple_tools_on_multiple_mcp_servers(
                    responses=responses,
                    urls=urls,
                    connections=connections,
                    output_type=output_type,
                    max_concurrent=max_concurrent,
                    *args,
                    **kwargs,
                )
            )
        except Exception as e:
            logger.error(
                f"Error in execute_multiple_tools_on_multiple_mcp_servers_sync: {str(e)}"
            )
            raise MCPExecutionError(
                f"Failed to execute multiple tools sync: {str(e)}"
            )
