import asyncio
import contextlib
import json
import os
import random
import traceback
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from typing import Any, Dict, List, Literal, Optional, Union

from litellm.types.utils import ChatCompletionMessageToolCall
from loguru import logger
from mcp import ClientSession
from mcp.client.sse import sse_client

try:
    from mcp.client.streamable_http import streamablehttp_client
except ImportError as e:
    logger.error(
        f"streamablehttp_client is not available. Import error: {str(e)}. "
        "Please ensure the MCP SDK is up to date with pip3 install -U mcp"
    )
    streamablehttp_client = None

from urllib.parse import urlparse

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

    def __init__(self, message: str, original_error: Optional[Exception] = None, traceback_str: Optional[str] = None):
        super().__init__(message)
        self.original_error = original_error
        self.traceback_str = traceback_str or self._get_traceback()
    
    def _get_traceback(self) -> str:
        """Get current traceback as string."""
        return ''.join(traceback.format_exc())
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.original_error:
            msg += f" (Original error: {str(self.original_error)})"
        if self.traceback_str:
            msg += f"\nTraceback:\n{self.traceback_str}"
        return msg


class MCPConnectionError(MCPError):
    """Raised when there are issues connecting to the MCP server."""

    def __init__(self, message: str, original_error: Optional[Exception] = None, server_url: Optional[str] = None, transport: Optional[str] = None):
        super().__init__(message, original_error)
        self.server_url = server_url
        self.transport = transport


class MCPToolError(MCPError):
    """Raised when there are issues with MCP tool operations."""

    def __init__(self, message: str, original_error: Optional[Exception] = None, tool_name: Optional[str] = None, server_url: Optional[str] = None):
        super().__init__(message, original_error)
        self.tool_name = tool_name
        self.server_url = server_url


class MCPValidationError(MCPError):
    """Raised when there are validation issues with MCP operations."""

    def __init__(self, message: str, original_error: Optional[Exception] = None, invalid_data: Optional[Any] = None):
        super().__init__(message, original_error)
        self.invalid_data = invalid_data


class MCPExecutionError(MCPError):
    """Raised when there are issues executing MCP operations."""

    def __init__(self, message: str, original_error: Optional[Exception] = None, operation: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, original_error)
        self.operation = operation
        self.context = context


def _log_error_with_traceback(error: Exception, context: str = "", additional_info: Optional[Dict[str, Any]] = None):
    """Helper function to log errors with full traceback and context."""
    error_msg = f"Error in {context}: {str(error)}"
    if additional_info:
        error_msg += f" | Additional info: {additional_info}"
    
    logger.error(error_msg)
    logger.error(f"Exception type: {type(error).__name__}")
    logger.error(f"Full traceback:\n{''.join(traceback.format_exc())}")
    
    # Log system info for debugging
    logger.error(f"Python version: {sys.version}")
    logger.error(f"Platform: {sys.platform}")


def _safe_json_parse(json_str: str, context: str = "") -> Optional[Dict[str, Any]]:
    """Safely parse JSON string with error handling."""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing failed in {context}: {str(e)}")
        logger.debug(f"Failed JSON string: {json_str[:200]}...")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing JSON in {context}: {str(e)}")
        _log_error_with_traceback(e, f"JSON parsing in {context}")
        return None


########################################################
# List MCP Tool functions
########################################################
def transform_mcp_tool_to_openai_tool(
    mcp_tool: MCPTool,
) -> ChatCompletionToolParam:
    """
    Convert an MCP tool to an OpenAI tool.
    Args:
        mcp_tool (MCPTool): The MCP tool object.
    Returns:
        ChatCompletionToolParam: The OpenAI-compatible tool parameter.
    Raises:
        MCPValidationError: If the MCP tool is invalid or missing required fields.
    """
    try:
        logger.info(
            f"Transforming MCP tool '{mcp_tool.name}' to OpenAI tool format."
        )
        
        if not hasattr(mcp_tool, 'name') or not mcp_tool.name:
            raise MCPValidationError(
                "MCP tool is missing required 'name' field",
                invalid_data=mcp_tool
            )
        
        if not hasattr(mcp_tool, 'inputSchema'):
            logger.warning(f"MCP tool '{mcp_tool.name}' missing inputSchema, using empty dict")
            input_schema = {}
        else:
            input_schema = mcp_tool.inputSchema
        
        return ChatCompletionToolParam(
            type="function",
            function=FunctionDefinition(
                name=mcp_tool.name,
                description=mcp_tool.description or "",
                parameters=input_schema,
                strict=False,
            ),
        )
    except Exception as e:
        _log_error_with_traceback(e, f"transforming MCP tool '{getattr(mcp_tool, 'name', 'unknown')}'")
        raise MCPValidationError(
            f"Failed to transform MCP tool to OpenAI format: {str(e)}",
            original_error=e,
            invalid_data=mcp_tool
        )


async def load_mcp_tools(
    session: ClientSession, format: Literal["mcp", "openai"] = "mcp"
) -> Union[List[MCPTool], List[ChatCompletionToolParam]]:
    """
    Load all available MCP tools from the session.
    Args:
        session (ClientSession): The MCP session to use.
        format (Literal["mcp", "openai"]): The format to convert the tools to.
    Returns:
        List of tools in the specified format.
    Raises:
        MCPToolError: If there are issues loading or transforming tools.
        MCPConnectionError: If there are connection issues.
    """
    try:
        logger.info(f"Loading MCP tools with format '{format}'.")
        
        if not session:
            raise MCPValidationError("Session object is required")
        
        tools = await session.list_tools()
        
        if not tools or not hasattr(tools, 'tools'):
            logger.warning("No tools returned from session.list_tools()")
            return []
        
        if format == "openai":
            openai_tools = []
            for i, tool in enumerate(tools.tools):
                try:
                    openai_tool = transform_mcp_tool_to_openai_tool(mcp_tool=tool)
                    openai_tools.append(openai_tool)
                except Exception as e:
                    logger.error(f"Failed to transform tool {i} '{getattr(tool, 'name', 'unknown')}': {str(e)}")
                    _log_error_with_traceback(e, f"transforming tool {i}")
                    # Continue with other tools instead of failing completely
                    continue
            
            if not openai_tools:
                logger.warning("No tools were successfully transformed to OpenAI format")
            
            return openai_tools
        
        return tools.tools
    except Exception as e:
        _log_error_with_traceback(e, "loading MCP tools")
        if "connection" in str(e).lower() or "timeout" in str(e).lower():
            raise MCPConnectionError(
                f"Failed to load MCP tools due to connection issue: {str(e)}",
                original_error=e
            )
        else:
            raise MCPToolError(
                f"Failed to load MCP tools: {str(e)}",
                original_error=e
            )


########################################################
# Call MCP Tool functions
########################################################


async def call_mcp_tool(
    session: ClientSession,
    call_tool_request_params: MCPCallToolRequestParams,
) -> MCPCallToolResult:
    """
    Call an MCP tool using the provided session and request parameters.
    Args:
        session (ClientSession): The MCP session to use.
        call_tool_request_params (MCPCallToolRequestParams): The tool call request params.
    Returns:
        MCPCallToolResult: The result of the tool call.
    Raises:
        MCPValidationError: If the request parameters are invalid.
        MCPToolError: If there are issues with the tool call.
        MCPConnectionError: If there are connection issues.
    """
    try:
        if not session:
            raise MCPValidationError("Session object is required")
        
        if not call_tool_request_params:
            raise MCPValidationError("Tool call request parameters are required")
        
        if not hasattr(call_tool_request_params, 'name') or not call_tool_request_params.name:
            raise MCPValidationError("Tool name is required in request parameters")
        
        logger.info(f"Calling MCP tool '{call_tool_request_params.name}' with arguments: {call_tool_request_params.arguments}")
        
        result = await session.call_tool(
            name=call_tool_request_params.name,
            arguments=call_tool_request_params.arguments,
        )
        
        if not result:
            logger.warning(f"Tool call to '{call_tool_request_params.name}' returned no result")
        
        return result
        
    except Exception as e:
        _log_error_with_traceback(e, f"calling MCP tool '{getattr(call_tool_request_params, 'name', 'unknown')}'")
        
        if "connection" in str(e).lower() or "timeout" in str(e).lower():
            raise MCPConnectionError(
                f"Failed to call MCP tool due to connection issue: {str(e)}",
                original_error=e,
                server_url=getattr(session, 'url', None)
            )
        elif "validation" in str(e).lower() or "invalid" in str(e).lower():
            raise MCPValidationError(
                f"Invalid tool call request: {str(e)}",
                original_error=e,
                invalid_data=call_tool_request_params
            )
        else:
            raise MCPToolError(
                f"Failed to call MCP tool '{getattr(call_tool_request_params, 'name', 'unknown')}': {str(e)}",
                original_error=e,
                tool_name=getattr(call_tool_request_params, 'name', None)
            )


def _get_function_arguments(function: FunctionDefinition) -> dict:
    """
    Helper to safely get and parse function arguments from a function definition.
    Args:
        function (FunctionDefinition): The function definition.
    Returns:
        dict: Parsed arguments as a dictionary.
    Raises:
        MCPValidationError: If function arguments cannot be parsed.
    """
    try:
        if not function:
            logger.warning("Function definition is None or empty")
            return {}
        
        if not isinstance(function, dict):
            logger.warning(f"Function definition is not a dict, got {type(function)}")
            return {}
        
        arguments = function.get("arguments", {})
        
        if isinstance(arguments, str):
            try:
                parsed_args = _safe_json_parse(arguments, "function arguments")
                if parsed_args is None:
                    logger.warning("Failed to parse function arguments JSON, using empty dict")
                    return {}
                return parsed_args
            except Exception as e:
                logger.error(f"Unexpected error parsing function arguments: {str(e)}")
                _log_error_with_traceback(e, "parsing function arguments")
                return {}
        
        if not isinstance(arguments, dict):
            logger.warning(f"Function arguments is not a dict, got {type(arguments)}")
            return {}
        
        return arguments
        
    except Exception as e:
        _log_error_with_traceback(e, "getting function arguments")
        logger.error(f"Failed to get function arguments: {str(e)}")
        return {}


def transform_openai_tool_call_request_to_mcp_tool_call_request(
    openai_tool: Union[ChatCompletionMessageToolCall, Dict],
) -> MCPCallToolRequestParams:
    """
    Convert an OpenAI ChatCompletionMessageToolCall to an MCP CallToolRequestParams.
    Args:
        openai_tool (Union[ChatCompletionMessageToolCall, Dict]): The OpenAI tool call request.
    Returns:
        MCPCallToolRequestParams: The MCP tool call request params.
    Raises:
        MCPValidationError: If the OpenAI tool call request is invalid.
    """
    try:
        if not openai_tool:
            raise MCPValidationError("OpenAI tool call request is required")
        
        if not isinstance(openai_tool, dict):
            raise MCPValidationError(f"OpenAI tool call request must be a dict, got {type(openai_tool)}")
        
        if "function" not in openai_tool:
            raise MCPValidationError("OpenAI tool call request must contain 'function' field")
        
        function = openai_tool["function"]
        
        if not isinstance(function, dict):
            raise MCPValidationError(f"Function field must be a dict, got {type(function)}")
        
        if "name" not in function:
            raise MCPValidationError("Function must contain 'name' field")
        
        function_name = function["name"]
        if not function_name:
            raise MCPValidationError("Function name cannot be empty")
        
        arguments = _get_function_arguments(function)
        
        logger.info(f"Transformed OpenAI tool call '{function_name}' to MCP format")
        
        return MCPCallToolRequestParams(
            name=function_name,
            arguments=arguments,
        )
        
    except Exception as e:
        _log_error_with_traceback(e, "transforming OpenAI tool call to MCP format")
        raise MCPValidationError(
            f"Failed to transform OpenAI tool call to MCP format: {str(e)}",
            original_error=e,
            invalid_data=openai_tool
        )


async def call_openai_tool(
    session: ClientSession,
    openai_tool: dict,
) -> MCPCallToolResult:
    """
    Call an OpenAI tool using MCP client.
    Args:
        session (ClientSession): The MCP session to use.
        openai_tool (dict): The OpenAI tool to call.
    Returns:
        MCPCallToolResult: The result of the MCP tool call.
    Raises:
        MCPValidationError: If the OpenAI tool is invalid.
        MCPToolError: If there are issues with the tool call.
        MCPConnectionError: If there are connection issues.
    """
    try:
        if not session:
            raise MCPValidationError("Session object is required")
        
        if not openai_tool:
            raise MCPValidationError("OpenAI tool is required")
        
        if not isinstance(openai_tool, dict):
            raise MCPValidationError(f"OpenAI tool must be a dict, got {type(openai_tool)}")
        
        logger.info(f"Calling OpenAI tool using MCP client: {openai_tool.get('function', {}).get('name', 'unknown')}")
        
        mcp_tool_call_request_params = (
            transform_openai_tool_call_request_to_mcp_tool_call_request(
                openai_tool=openai_tool,
            )
        )
        
        result = await call_mcp_tool(
            session=session,
            call_tool_request_params=mcp_tool_call_request_params,
        )
        
        logger.info(f"Successfully called OpenAI tool '{mcp_tool_call_request_params.name}' via MCP")
        return result
        
    except Exception as e:
        _log_error_with_traceback(e, "calling OpenAI tool via MCP")
        
        if isinstance(e, (MCPValidationError, MCPToolError, MCPConnectionError)):
            raise
        
        # Re-raise with appropriate MCP exception type
        if "connection" in str(e).lower() or "timeout" in str(e).lower():
            raise MCPConnectionError(
                f"Failed to call OpenAI tool due to connection issue: {str(e)}",
                original_error=e,
                server_url=getattr(session, 'url', None)
            )
        elif "validation" in str(e).lower() or "invalid" in str(e).lower():
            raise MCPValidationError(
                f"Invalid OpenAI tool call: {str(e)}",
                original_error=e,
                invalid_data=openai_tool
            )
        else:
            raise MCPToolError(
                f"Failed to call OpenAI tool: {str(e)}",
                original_error=e,
                tool_name=openai_tool.get('function', {}).get('name', 'unknown') if isinstance(openai_tool, dict) else None
            )


def retry_with_backoff(retries=3, backoff_in_seconds=1):
    """
    Decorator for retrying async functions with exponential backoff.
    Args:
        retries (int): Number of retry attempts.
        backoff_in_seconds (int): Initial backoff time in seconds.
    Returns:
        Decorated async function with retry logic.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == retries:
                        logger.error(
                            f"Failed after {retries} retries. Final error: {str(e)}"
                        )
                        _log_error_with_traceback(e, f"final retry attempt {attempt + 1}")
                        raise
                    
                    sleep_time = (
                        backoff_in_seconds * 2**attempt
                        + random.uniform(0, 1)
                    )
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {sleep_time:.2f}s. "
                        f"Exception type: {type(e).__name__}"
                    )
                    
                    # Log detailed error info for debugging
                    if attempt == 0:  # Only log full traceback on first failure
                        _log_error_with_traceback(e, f"retry attempt {attempt + 1}")
                    
                    await asyncio.sleep(sleep_time)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


@contextlib.contextmanager
def get_or_create_event_loop():
    """
    Context manager to handle event loop creation and cleanup.
    Yields:
        asyncio.AbstractEventLoop: The event loop to use.
    Ensures the event loop is properly closed if created here.
    """
    loop = None
    created_new_loop = False
    
    try:
        # Try to get the current event loop
        try:
            loop = asyncio.get_running_loop()
            logger.debug("Using already running event loop")
        except RuntimeError:
            # No running loop, try to get the current event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Loop is running in another thread, create a new one
                    logger.debug("Current event loop is running, creating new one")
                    loop = asyncio.new_event_loop()
                    created_new_loop = True
                else:
                    logger.debug("Using existing event loop")
            except RuntimeError:
                # No event loop exists, create a new one
                logger.debug("No event loop exists, creating new one")
                loop = asyncio.new_event_loop()
                created_new_loop = True
        
        # Set the loop if we created a new one
        if created_new_loop:
            asyncio.set_event_loop(loop)
        
        yield loop
        
    finally:
        # Only close the loop if we created it and it's not the main event loop
        if created_new_loop and loop and not loop.is_running():
            try:
                if not loop.is_closed():
                    loop.close()
                    logger.debug("Closed newly created event loop")
            except Exception as e:
                logger.warning(f"Error closing event loop: {str(e)}")


def run_in_event_loop(coro, loop=None):
    """
    Safely run a coroutine in an event loop, handling various loop states.
    
    Args:
        coro: The coroutine to run
        loop: Optional event loop to use
        
    Returns:
        The result of the coroutine
        
    Raises:
        RuntimeError: If unable to run the coroutine
    """
    if loop is None:
        try:
            # Try to get the current running loop
            loop = asyncio.get_running_loop()
            logger.debug("Using currently running event loop")
            # If we're in a running loop, we need to create a task
            # This is a fallback for when run_until_complete can't be used
            raise RuntimeError("Cannot use run_until_complete in running event loop")
        except RuntimeError:
            # No running loop, get or create one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Loop is running in another thread, create a new one
                    logger.debug("Creating new event loop for thread")
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(coro)
                    finally:
                        loop.close()
                        asyncio.set_event_loop(None)
                else:
                    # Use existing loop
                    logger.debug("Using existing event loop")
                    return loop.run_until_complete(coro)
            except RuntimeError:
                # No event loop exists, create a new one
                logger.debug("Creating new event loop")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)


def run_in_event_loop_with_fallback(coro):
    """
    Enhanced version that tries multiple approaches to run a coroutine.
    
    Args:
        coro: The coroutine to run
        
    Returns:
        The result of the coroutine
        
    Raises:
        RuntimeError: If all approaches fail
    """
    try:
        # First try the standard approach
        return run_in_event_loop(coro)
    except RuntimeError as e:
        if "Cannot use run_until_complete in running event loop" in str(e):
            # We're in a running event loop, try to use asyncio.create_task
            try:
                logger.debug("Attempting to use asyncio.create_task as fallback")
                loop = asyncio.get_running_loop()
                
                # Create a future to get the result
                future = asyncio.Future()
                
                async def wrapper():
                    try:
                        result = await coro
                        future.set_result(result)
                    except Exception as exc:
                        future.set_exception(exc)
                
                # Create and run the task
                task = loop.create_task(wrapper())
                
                # Wait for completion with timeout
                try:
                    # Use asyncio.wait_for with a reasonable timeout
                    loop.create_task(asyncio.wait_for(future, timeout=300))  # 5 minute timeout
                    return future.result()
                except asyncio.TimeoutError:
                    task.cancel()
                    raise RuntimeError("Coroutine execution timed out")
                except Exception as exc:
                    task.cancel()
                    raise exc
                    
            except Exception as fallback_error:
                logger.error(f"Fallback approach also failed: {str(fallback_error)}")
                _log_error_with_traceback(fallback_error, "fallback event loop execution")
                raise RuntimeError(f"All event loop execution approaches failed: {str(e)} -> {str(fallback_error)}")
        else:
            # Re-raise the original error
            raise


def connect_to_mcp_server(connection: MCPConnection = None):
    """
    Connect to an MCP server using the provided connection configuration.
    Args:
        connection (MCPConnection): The connection configuration object.
    Returns:
        tuple: (headers, timeout, transport, url)
    Raises:
        MCPValidationError: If the connection object is invalid.
    """
    try:
        logger.info(
            "Connecting to MCP server using MCPConnection object."
        )
        
        if not isinstance(connection, MCPConnection):
            raise MCPValidationError(
                f"Invalid connection type provided to connect_to_mcp_server. "
                f"Expected MCPConnection, got {type(connection)}"
            )
        
        if not connection.url:
            raise MCPValidationError("Connection URL is required")
        
        headers = dict(connection.headers or {})
        if connection.authorization_token:
            headers["Authorization"] = (
                f"Bearer {connection.authorization_token}"
            )
        
        timeout = connection.timeout or 5
        transport = connection.transport or "sse"
        
        logger.info(f"Connection configured: URL={connection.url}, Transport={transport}, Timeout={timeout}")
        
        return (
            headers,
            timeout,
            transport,
            connection.url,
        )
        
    except Exception as e:
        _log_error_with_traceback(e, "connecting to MCP server")
        if isinstance(e, MCPValidationError):
            raise
        raise MCPValidationError(
            f"Failed to configure MCP connection: {str(e)}",
            original_error=e,
            invalid_data=connection
        )


def get_mcp_client(transport, url, headers=None, timeout=5, **kwargs):
    """
    Helper to select the correct MCP client context manager based on transport.
    Supports 'sse' (default) and 'streamable_http'.
    Args:
        transport (str): The transport type ('sse' or 'streamable_http').
        url (str): The server URL.
        headers (dict): Optional headers.
        timeout (int): Timeout in seconds.
        **kwargs: Additional arguments.
    Returns:
        Context manager for the selected client.
    Raises:
        ImportError: If streamablehttp_client is not available when requested.
        MCPValidationError: If transport type is invalid.
    """
    try:
        logger.info(
            f"Getting MCP client for transport '{transport}' and url '{url}'."
        )
        
        if not transport:
            raise MCPValidationError("Transport type is required")
        
        if not url:
            raise MCPValidationError("Server URL is required")
        
        if transport == "streamable_http":
            if streamablehttp_client is None:
                raise ImportError(
                    "streamablehttp_client is not available. Please ensure the MCP SDK is up to date."
                )
            
            logger.info(f"Using streamable HTTP client for {url}")
            return streamablehttp_client(
                url, headers=headers, timeout=timeout, **kwargs
            )
        elif transport == "sse":
            logger.info(f"Using SSE client for {url}")
            return sse_client(
                url, headers=headers, timeout=timeout, **kwargs
            )
        else:
            raise MCPValidationError(
                f"Unsupported transport type: {transport}. "
                f"Supported types: 'sse', 'streamable_http'"
            )
            
    except Exception as e:
        _log_error_with_traceback(e, f"getting MCP client for transport '{transport}'")
        
        if isinstance(e, (ImportError, MCPValidationError)):
            raise
        
        raise MCPValidationError(
            f"Failed to get MCP client: {str(e)}",
            original_error=e,
            invalid_data={"transport": transport, "url": url}
        )


def auto_detect_transport(url: str) -> str:
    """
    Guess the MCP transport based on the URL scheme and path.
    Does not make any network requests.
    Returns one of: 'streamable_http', 'sse', or 'stdio'.
    Args:
        url (str): The server URL.
    Returns:
        str: The detected transport type.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme in ("http", "https"):
        logger.info(
            f"Automatically selected 'streamable_http' transport for {url}"
        )
        return "streamable_http"
    elif scheme in ("ws", "wss"):
        logger.info(
            f"Automatically selected 'sse' transport for {url}"
        )
        return "sse"  # or 'websocket' if you support it
    elif "stdio" in url or scheme == "":
        logger.info(
            f"Automatically selected 'stdio' transport for {url}"
        )
        return "stdio"
    else:
        logger.info(f"Defaulting to 'sse' transport for {url}")
        return "sse"


@retry_with_backoff(retries=3)
async def aget_mcp_tools(
    server_path: Optional[str] = None,
    format: str = "openai",
    connection: Optional[MCPConnection] = None,
    transport: Optional[str] = None,
    *args,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Fetch available MCP tools from the server with retry logic.
    Args:
        server_path (str): Path to the MCP server script.
        format (str): Format to return tools in ('openai' or 'mcp').
        connection (Optional[MCPConnection]): Optional connection object.
        transport (Optional[str]): Transport type. If None, auto-detects.
    Returns:
        List[Dict[str, Any]]: List of available MCP tools in OpenAI format.
    Raises:
        MCPValidationError: If server_path is invalid.
        MCPConnectionError: If connection to server fails.
        MCPToolError: If there are issues loading tools.
    """
    try:
        logger.info(
            f"aget_mcp_tools called for server_path: {server_path}"
        )
        
        if not server_path and not connection:
            raise MCPValidationError("Either server_path or connection must be provided")
        
        if transport is None:
            transport = auto_detect_transport(server_path or "")
            logger.info(f"Auto-detected transport: {transport}")
        
        if exists(connection):
            try:
                headers, timeout, transport_from_conn, url = (
                    connect_to_mcp_server(connection)
                )
                if transport_from_conn:
                    transport = transport_from_conn
                    logger.info(f"Using transport from connection: {transport}")
            except Exception as e:
                _log_error_with_traceback(e, "processing MCP connection")
                raise MCPConnectionError(
                    f"Failed to process MCP connection: {str(e)}",
                    original_error=e,
                    server_url=getattr(connection, 'url', None),
                    transport=transport
                )
        else:
            headers, timeout, _transport, _url = (
                None,
                5,
                None,
                server_path,
            )
            url = server_path
        
        logger.info(
            f"Fetching MCP tools from server: {url} using transport: {transport}"
        )
        
        try:
            async with get_mcp_client(
                transport,
                url=url,
                headers=headers,
                timeout=timeout,
                *args,
                **kwargs,
            ) as ctx:
                if len(ctx) == 2:
                    read, write = ctx
                else:
                    read, write, *_ = ctx
                
                async with ClientSession(read, write) as session:
                    try:
                        await session.initialize()
                        logger.info("MCP session initialized successfully")
                        
                        tools = await load_mcp_tools(
                            session=session, format=format
                        )
                        
                        logger.info(
                            f"Successfully fetched {len(tools)} tools"
                        )
                        return tools
                        
                    except Exception as e:
                        _log_error_with_traceback(e, "loading MCP tools in session")
                        raise MCPToolError(
                            f"Failed to load MCP tools in session: {str(e)}",
                            original_error=e,
                            server_url=url
                        )
                        
        except Exception as e:
            _log_error_with_traceback(e, f"MCP client connection to {url}")
            raise MCPConnectionError(
                f"Failed to connect to MCP server: {str(e)}",
                original_error=e,
                server_url=url,
                transport=transport
            )
            
    except Exception as e:
        _log_error_with_traceback(e, "aget_mcp_tools")
        
        if isinstance(e, (MCPValidationError, MCPConnectionError, MCPToolError)):
            raise
        
        # Re-raise with appropriate MCP exception type
        if "connection" in str(e).lower() or "timeout" in str(e).lower():
            raise MCPConnectionError(
                f"Failed to fetch MCP tools due to connection issue: {str(e)}",
                original_error=e,
                server_url=server_path
            )
        elif "validation" in str(e).lower() or "invalid" in str(e).lower():
            raise MCPValidationError(
                f"Invalid parameters for fetching MCP tools: {str(e)}",
                original_error=e,
                invalid_data={"server_path": server_path, "connection": connection}
            )
        else:
            raise MCPToolError(
                f"Failed to fetch MCP tools: {str(e)}",
                original_error=e,
                server_url=server_path
            )


def get_mcp_tools_sync(
    server_path: Optional[str] = None,
    format: str = "openai",
    connection: Optional[MCPConnection] = None,
    transport: Optional[str] = None,
    *args,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Synchronous version of get_mcp_tools that handles event loop management.
    Args:
        server_path (str): Path to the MCP server script.
        format (str): Format to return tools in ('openai' or 'mcp').
        connection (Optional[MCPConnection]): Optional connection object.
        transport (Optional[str]): Transport type. If None, auto-detects.
    Returns:
        List[Dict[str, Any]]: List of available MCP tools in OpenAI format.
    Raises:
        MCPValidationError: If server_path is invalid.
        MCPConnectionError: If connection to server fails.
        MCPExecutionError: If event loop management fails.
        MCPToolError: If there are issues loading tools.
    """
    try:
        logger.info(
            f"get_mcp_tools_sync called for server_path: {server_path}"
        )
        
        if transport is None:
            transport = auto_detect_transport(server_path or "")
            logger.info(f"Auto-detected transport: {transport}")
        
        # Use the enhanced event loop runner with fallback
        return run_in_event_loop_with_fallback(
            aget_mcp_tools(
                server_path=server_path,
                format=format,
                connection=connection,
                transport=transport,
                *args,
                **kwargs,
            )
        )
                
    except Exception as e:
        _log_error_with_traceback(e, "get_mcp_tools_sync")
        
        if isinstance(e, (MCPValidationError, MCPConnectionError, MCPExecutionError, MCPToolError)):
            raise
        
        # Re-raise with appropriate MCP exception type
        if "connection" in str(e).lower() or "timeout" in str(e).lower():
            raise MCPConnectionError(
                f"Failed to fetch MCP tools due to connection issue: {str(e)}",
                original_error=e,
                server_url=server_path
            )
        elif "validation" in str(e).lower() or "invalid" in str(e).lower():
            raise MCPValidationError(
                f"Invalid parameters for fetching MCP tools: {str(e)}",
                original_error=e,
                invalid_data={"server_path": server_path, "connection": connection}
            )
        else:
            raise MCPExecutionError(
                f"Failed to execute MCP tools sync: {str(e)}",
                original_error=e,
                operation="get_mcp_tools_sync",
                context={"server_path": server_path, "connection": connection}
            )


def _fetch_tools_for_server(
    url: str,
    connection: Optional[MCPConnection] = None,
    format: str = "openai",
    transport: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Helper function to fetch tools for a single server.
    Args:
        url (str): The server URL.
        connection (Optional[MCPConnection]): Optional connection object.
        format (str): Format to return tools in.
        transport (Optional[str]): Transport type. If None, auto-detects.
    Returns:
        List[Dict[str, Any]]: List of available MCP tools.
    Raises:
        MCPExecutionError: If tool fetching fails.
    """
    try:
        logger.info(f"_fetch_tools_for_server called for url: {url}")
        
        if not url:
            raise MCPValidationError("Server URL is required")
        
        if transport is None:
            transport = auto_detect_transport(url)
            logger.info(f"Auto-detected transport for {url}: {transport}")
        
        tools = get_mcp_tools_sync(
            server_path=url,
            connection=connection,
            format=format,
            transport=transport,
        )
        
        logger.info(f"Successfully fetched {len(tools)} tools from {url}")
        return tools
        
    except Exception as e:
        _log_error_with_traceback(e, f"fetching tools from server {url}")
        
        if isinstance(e, (MCPValidationError, MCPConnectionError, MCPToolError)):
            # Re-raise as MCPExecutionError to maintain consistent error handling
            raise MCPExecutionError(
                f"Failed to fetch tools from server {url}: {str(e)}",
                original_error=e,
                operation="fetch_tools_for_server",
                context={"url": url, "connection": connection, "format": format, "transport": transport}
            )
        else:
            raise MCPExecutionError(
                f"Failed to fetch tools from server {url}: {str(e)}",
                original_error=e,
                operation="fetch_tools_for_server",
                context={"url": url, "connection": connection, "format": format, "transport": transport}
            )


def get_tools_for_multiple_mcp_servers(
    urls: List[str],
    connections: List[MCPConnection] = None,
    format: str = "openai",
    output_type: Literal["json", "dict", "str"] = "str",
    max_workers: Optional[int] = None,
    transport: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get tools for multiple MCP servers concurrently using ThreadPoolExecutor.
    Args:
        urls (List[str]): List of server URLs to fetch tools from.
        connections (List[MCPConnection]): Optional list of MCPConnection objects.
        format (str): Format to return tools in.
        output_type (Literal): Output format type.
        max_workers (Optional[int]): Max worker threads.
        transport (Optional[str]): Transport type. If None, auto-detects per URL.
    Returns:
        List[Dict[str, Any]]: Combined list of tools from all servers.
    Raises:
        MCPValidationError: If parameters are invalid.
        MCPExecutionError: If tool fetching fails.
    """
    try:
        logger.info(
            f"get_tools_for_multiple_mcp_servers called for {len(urls)} urls."
        )
        
        if not urls:
            raise MCPValidationError("At least one server URL is required")
        
        if not isinstance(urls, list):
            raise MCPValidationError(f"URLs must be a list, got {type(urls)}")
        
        if connections and not isinstance(connections, list):
            raise MCPValidationError(f"Connections must be a list, got {type(connections)}")
        
        if connections and len(connections) != len(urls):
            logger.warning(f"Number of connections ({len(connections)}) doesn't match number of URLs ({len(urls)})")
        
        tools = []
        max_workers = (
            min(32, os.cpu_count() + 4)
            if max_workers is None
            else max_workers
        )
        
        logger.info(f"Using {max_workers} worker threads for concurrent tool fetching")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            if exists(connections):
                future_to_url = {
                    executor.submit(
                        _fetch_tools_for_server,
                        url,
                        connection,
                        format,
                        transport,
                    ): url
                    for url, connection in zip(urls, connections)
                }
            else:
                future_to_url = {
                    executor.submit(
                        _fetch_tools_for_server,
                        url,
                        None,
                        format,
                        transport,
                    ): url
                    for url in urls
                }
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    server_tools = future.result()
                    if server_tools:
                        tools.extend(server_tools)
                        logger.info(f"Successfully fetched {len(server_tools)} tools from {url}")
                    else:
                        logger.warning(f"No tools returned from {url}")
                except Exception as e:
                    _log_error_with_traceback(e, f"fetching tools from {url}")
                    raise MCPExecutionError(
                        f"Failed to fetch tools from {url}: {str(e)}",
                        original_error=e,
                        operation="get_tools_for_multiple_mcp_servers",
                        context={"url": url, "connections": connections, "format": format, "transport": transport}
                    )
        
        logger.info(f"Successfully fetched total of {len(tools)} tools from {len(urls)} servers")
        return tools
        
    except Exception as e:
        _log_error_with_traceback(e, "get_tools_for_multiple_mcp_servers")
        
        if isinstance(e, (MCPValidationError, MCPExecutionError)):
            raise
        
        # Re-raise with appropriate MCP exception type
        if "validation" in str(e).lower() or "invalid" in str(e).lower():
            raise MCPValidationError(
                f"Invalid parameters for multiple MCP servers: {str(e)}",
                original_error=e,
                invalid_data={"urls": urls, "connections": connections}
            )
        else:
            raise MCPExecutionError(
                f"Failed to get tools for multiple MCP servers: {str(e)}",
                original_error=e,
                operation="get_tools_for_multiple_mcp_servers",
                context={"urls": urls, "connections": connections, "format": format, "transport": transport}
            )


async def _execute_tool_call_simple(
    response: any = None,
    server_path: str = None,
    connection: Optional[MCPConnection] = None,
    output_type: Literal["json", "dict", "str"] = "str",
    transport: Optional[str] = None,
    *args,
    **kwargs,
):
    """
    Execute a tool call using the MCP client, supporting both SSE and streamable HTTP.
    Args:
        response (any): The tool call request.
        server_path (str): The server URL.
        connection (Optional[MCPConnection]): Optional connection object.
        output_type (Literal): Output format type.
        transport (Optional[str]): Transport type. If None, auto-detects.
    Returns:
        The tool call result in the specified output format.
    Raises:
        MCPExecutionError: If tool execution fails.
        MCPConnectionError: If connection fails.
        MCPValidationError: If parameters are invalid.
    """
    try:
        logger.info(
            f"_execute_tool_call_simple called for server_path: {server_path}"
        )
        
        if not response:
            raise MCPValidationError("Tool call response is required")
        
        if not server_path:
            raise MCPValidationError("Server path is required")
        
        if transport is None:
            transport = auto_detect_transport(server_path)
            logger.info(f"Auto-detected transport: {transport}")
        
        if exists(connection):
            try:
                headers, timeout, transport_from_conn, url = (
                    connect_to_mcp_server(connection)
                )
                if transport_from_conn:
                    transport = transport_from_conn
                    logger.info(f"Using transport from connection: {transport}")
            except Exception as e:
                _log_error_with_traceback(e, "processing MCP connection in _execute_tool_call_simple")
                raise MCPConnectionError(
                    f"Failed to process MCP connection: {str(e)}",
                    original_error=e,
                    server_url=getattr(connection, 'url', None),
                    transport=transport
                )
        else:
            headers, timeout, _transport, url = (
                None,
                5,
                "sse",
                server_path,
            )
        
        try:
            async with get_mcp_client(
                transport,
                url=url,
                headers=headers,
                timeout=timeout,
                *args,
                **kwargs,
            ) as ctx:
                if len(ctx) == 2:
                    read, write = ctx
                else:
                    read, write, *_ = ctx
                
                async with ClientSession(read, write) as session:
                    try:
                        await session.initialize()
                        logger.info("MCP session initialized successfully for tool execution")
                        
                        call_result = await call_openai_tool(
                            session=session, openai_tool=response
                        )
                        
                        if not call_result:
                            logger.warning("Tool call returned no result")
                        
                        # Format output based on output_type
                        try:
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
                                                    f"item: {item}"
                                                )
                                    else:
                                        formatted_lines.append(
                                            f"{key}: {value}"
                                        )
                                out = "\n".join(formatted_lines)
                            else:
                                out = call_result.model_dump()
                            
                            logger.info(
                                f"Tool call executed successfully for {server_path}"
                            )
                            return out
                            
                        except Exception as e:
                            _log_error_with_traceback(e, "formatting tool call result")
                            logger.warning(f"Failed to format result as {output_type}, returning raw result")
                            return call_result
                            
                    except Exception as e:
                        _log_error_with_traceback(e, "executing tool call in session")
                        tool_name = "unknown"
                        if isinstance(response, dict) and "function" in response:
                            tool_name = response["function"].get("name", "unknown")
                        
                        raise MCPExecutionError(
                            f"Tool execution failed for tool '{tool_name}' on server '{url}': {str(e)}",
                            original_error=e,
                            operation="tool_execution",
                            context={"tool_name": tool_name, "server_url": url, "output_type": output_type}
                        )
                        
        except Exception as e:
            _log_error_with_traceback(e, f"MCP client connection to {url}")
            raise MCPConnectionError(
                f"Failed to connect to MCP server '{url}' using transport '{transport}': {str(e)}",
                original_error=e,
                server_url=url,
                transport=transport
            )
            
    except Exception as e:
        _log_error_with_traceback(e, "_execute_tool_call_simple")
        
        if isinstance(e, (MCPValidationError, MCPConnectionError, MCPExecutionError)):
            raise
        
        # Re-raise with appropriate MCP exception type
        if "connection" in str(e).lower() or "timeout" in str(e).lower():
            raise MCPConnectionError(
                f"Failed to execute tool call due to connection issue: {str(e)}",
                original_error=e,
                server_url=server_path
            )
        elif "validation" in str(e).lower() or "invalid" in str(e).lower():
            raise MCPValidationError(
                f"Invalid parameters for tool call execution: {str(e)}",
                original_error=e,
                invalid_data={"response": response, "server_path": server_path}
            )
        else:
            raise MCPExecutionError(
                f"Failed to execute tool call: {str(e)}",
                original_error=e,
                operation="_execute_tool_call_simple",
                context={"response": response, "server_path": server_path}
            )


async def execute_tool_call_simple(
    response: any = None,
    server_path: str = None,
    connection: Optional[MCPConnection] = None,
    output_type: Literal["json", "dict", "str", "formatted"] = "str",
    transport: Optional[str] = None,
    *args,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    High-level async function to execute a tool call on an MCP server.
    Args:
        response (any): The tool call request.
        server_path (str): The server URL.
        connection (Optional[MCPConnection]): Optional connection object.
        output_type (Literal): Output format type.
        transport (Optional[str]): Transport type. If None, auto-detects.
    Returns:
        The tool call result in the specified output format.
    Raises:
        MCPValidationError: If parameters are invalid.
        MCPConnectionError: If connection fails.
        MCPExecutionError: If tool execution fails.
    """
    try:
        logger.info(
            f"execute_tool_call_simple called for server_path: {server_path}"
        )
        
        if not response:
            raise MCPValidationError("Tool call response is required")
        
        if not server_path:
            raise MCPValidationError("Server path is required")
        
        if transport is None:
            transport = auto_detect_transport(server_path)
            logger.info(f"Auto-detected transport: {transport}")
        
        # Handle string response by parsing JSON
        if isinstance(response, str):
            try:
                parsed_response = _safe_json_parse(response, "tool call response")
                if parsed_response is None:
                    raise MCPValidationError("Failed to parse response JSON string")
                response = parsed_response
                logger.info("Successfully parsed JSON string response")
            except Exception as e:
                _log_error_with_traceback(e, "parsing JSON response string")
                raise MCPValidationError(
                    f"Failed to parse response JSON string: {str(e)}",
                    original_error=e,
                    invalid_data=response
                )
        
        if not isinstance(response, dict):
            raise MCPValidationError(f"Response must be a dict after parsing, got {type(response)}")
        
        result = await _execute_tool_call_simple(
            response=response,
            server_path=server_path,
            connection=connection,
            output_type=output_type,
            transport=transport,
            *args,
            **kwargs,
        )
        
        return result
        
    except Exception as e:
        _log_error_with_traceback(e, "execute_tool_call_simple")
        
        if isinstance(e, (MCPValidationError, MCPConnectionError, MCPExecutionError)):
            raise
        
        # Re-raise with appropriate MCP exception type
        if "connection" in str(e).lower() or "timeout" in str(e).lower():
            raise MCPConnectionError(
                f"Failed to execute tool call due to connection issue: {str(e)}",
                original_error=e,
                server_url=server_path
            )
        elif "validation" in str(e).lower() or "invalid" in str(e).lower():
            raise MCPValidationError(
                f"Invalid parameters for tool call execution: {str(e)}",
                original_error=e,
                invalid_data={"response": response, "server_path": server_path}
            )
        else:
            raise MCPExecutionError(
                f"Failed to execute tool call: {str(e)}",
                original_error=e,
                operation="execute_tool_call_simple",
                context={"response": response, "server_path": server_path}
            )


def _create_server_tool_mapping(
    urls: List[str],
    connections: List[MCPConnection] = None,
    format: str = "openai",
    transport: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Create a mapping of function names to server information for all MCP servers.
    Args:
        urls (List[str]): List of server URLs.
        connections (List[MCPConnection]): Optional list of MCPConnection objects.
        format (str): Format to fetch tools in.
        transport (Optional[str]): Transport type. If None, auto-detects per URL.
    Returns:
        Dict[str, Dict[str, Any]]: Mapping of function names to server info.
    Raises:
        MCPValidationError: If parameters are invalid.
        MCPExecutionError: If tool mapping creation fails.
    """
    try:
        if not urls:
            raise MCPValidationError("At least one server URL is required")
        
        if not isinstance(urls, list):
            raise MCPValidationError(f"URLs must be a list, got {type(urls)}")
        
        if connections and not isinstance(connections, list):
            raise MCPValidationError(f"Connections must be a list, got {type(connections)}")
        
        logger.info(f"Creating server tool mapping for {len(urls)} servers")
        
        server_tool_mapping = {}
        for i, url in enumerate(urls):
            try:
                connection = (
                    connections[i]
                    if connections and i < len(connections)
                    else None
                )
                
                logger.info(f"Fetching tools from server {i+1}/{len(urls)}: {url}")
                
                tools = get_mcp_tools_sync(
                    server_path=url,
                    connection=connection,
                    format=format,
                    transport=transport,
                )
                
                if not tools:
                    logger.warning(f"No tools returned from server {url}")
                    continue
                
                for tool in tools:
                    try:
                        if isinstance(tool, dict) and "function" in tool:
                            function_name = tool["function"]["name"]
                            server_tool_mapping[function_name] = {
                                "url": url,
                                "connection": connection,
                                "tool": tool,
                                "server_index": i,
                            }
                        elif hasattr(tool, "name"):
                            server_tool_mapping[tool.name] = {
                                "url": url,
                                "connection": connection,
                                "tool": tool,
                                "server_index": i,
                            }
                        else:
                            logger.warning(f"Tool from {url} has no recognizable name format: {tool}")
                    except Exception as e:
                        logger.error(f"Failed to process tool from {url}: {str(e)}")
                        _log_error_with_traceback(e, f"processing tool from {url}")
                        continue
                        
            except Exception as e:
                logger.warning(
                    f"Failed to fetch tools from server {url}: {str(e)}"
                )
                _log_error_with_traceback(e, f"fetching tools from server {url}")
                continue
        
        logger.info(f"Successfully created mapping with {len(server_tool_mapping)} unique functions")
        return server_tool_mapping
        
    except Exception as e:
        _log_error_with_traceback(e, "creating server tool mapping")
        
        if isinstance(e, (MCPValidationError, MCPExecutionError)):
            raise
        
        # Re-raise with appropriate MCP exception type
        if "validation" in str(e).lower() or "invalid" in str(e).lower():
            raise MCPValidationError(
                f"Invalid parameters for server tool mapping: {str(e)}",
                original_error=e,
                invalid_data={"urls": urls, "connections": connections}
            )
        else:
            raise MCPExecutionError(
                f"Failed to create server tool mapping: {str(e)}",
                original_error=e,
                operation="create_server_tool_mapping",
                context={"urls": urls, "connections": connections, "format": format, "transport": transport}
            )


async def _create_server_tool_mapping_async(
    urls: List[str],
    connections: List[MCPConnection] = None,
    format: str = "openai",
    transport: str = "sse",
) -> Dict[str, Dict[str, Any]]:
    """
    Async version: Create a mapping of function names to server information for all MCP servers.
    Args:
        urls (List[str]): List of server URLs.
        connections (List[MCPConnection]): Optional list of MCPConnection objects.
        format (str): Format to fetch tools in.
        transport (str): Transport type.
    Returns:
        Dict[str, Dict[str, Any]]: Mapping of function names to server info.
    Raises:
        MCPValidationError: If parameters are invalid.
        MCPExecutionError: If tool mapping creation fails.
    """
    try:
        if not urls:
            raise MCPValidationError("At least one server URL is required")
        
        if not isinstance(urls, list):
            raise MCPValidationError(f"URLs must be a list, got {type(urls)}")
        
        if connections and not isinstance(connections, list):
            raise MCPValidationError(f"Connections must be a list, got {type(connections)}")
        
        logger.info(f"Creating async server tool mapping for {len(urls)} servers")
        
        server_tool_mapping = {}
        for i, url in enumerate(urls):
            try:
                connection = (
                    connections[i]
                    if connections and i < len(connections)
                    else None
                )
                
                logger.info(f"Fetching tools from server {i+1}/{len(urls)}: {url}")
                
                tools = await aget_mcp_tools(
                    server_path=url,
                    connection=connection,
                    format=format,
                    transport=transport,
                )
                
                if not tools:
                    logger.warning(f"No tools returned from server {url}")
                    continue
                
                for tool in tools:
                    try:
                        if isinstance(tool, dict) and "function" in tool:
                            function_name = tool["function"]["name"]
                            server_tool_mapping[function_name] = {
                                "url": url,
                                "connection": connection,
                                "tool": tool,
                                "server_index": i,
                            }
                        elif hasattr(tool, "name"):
                            server_tool_mapping[tool.name] = {
                                "url": url,
                                "connection": connection,
                                "tool": tool,
                                "server_index": i,
                            }
                        else:
                            logger.warning(f"Tool from {url} has no recognizable name format: {tool}")
                    except Exception as e:
                        logger.error(f"Failed to process tool from {url}: {str(e)}")
                        _log_error_with_traceback(e, f"processing tool from {url}")
                        continue
                        
            except Exception as e:
                logger.warning(
                    f"Failed to fetch tools from server {url}: {str(e)}"
                )
                _log_error_with_traceback(e, f"fetching tools from server {url}")
                continue
        
        logger.info(f"Successfully created async mapping with {len(server_tool_mapping)} unique functions")
        return server_tool_mapping
        
    except Exception as e:
        _log_error_with_traceback(e, "creating async server tool mapping")
        
        if isinstance(e, (MCPValidationError, MCPExecutionError)):
            raise
        
        # Re-raise with appropriate MCP exception type
        if "validation" in str(e).lower() or "invalid" in str(e).lower():
            raise MCPValidationError(
                f"Invalid parameters for async server tool mapping: {str(e)}",
                original_error=e,
                invalid_data={"urls": urls, "connections": connections}
            )
        else:
            raise MCPExecutionError(
                f"Failed to create async server tool mapping: {str(e)}",
                original_error=e,
                operation="create_server_tool_mapping_async",
                context={"urls": urls, "connections": connections, "format": format, "transport": transport}
            )


async def _execute_tool_on_server(
    tool_call: Dict[str, Any],
    server_info: Dict[str, Any],
    output_type: Literal["json", "dict", "str", "formatted"] = "str",
    transport: str = "sse",
) -> Dict[str, Any]:
    """
    Execute a single tool call on a specific server.
    Args:
        tool_call (Dict[str, Any]): The tool call to execute.
        server_info (Dict[str, Any]): Server information from the mapping.
        output_type (Literal): Output format type.
        transport (str): Transport type.
    Returns:
        Dict[str, Any]: Execution result with server metadata.
    Raises:
        MCPValidationError: If parameters are invalid.
        MCPExecutionError: If tool execution fails.
    """
    try:
        if not tool_call:
            raise MCPValidationError("Tool call is required")
        
        if not server_info:
            raise MCPValidationError("Server info is required")
        
        if not isinstance(tool_call, dict):
            raise MCPValidationError(f"Tool call must be a dict, got {type(tool_call)}")
        
        if not isinstance(server_info, dict):
            raise MCPValidationError(f"Server info must be a dict, got {type(server_info)}")
        
        required_fields = ["url", "server_index"]
        for field in required_fields:
            if field not in server_info:
                raise MCPValidationError(f"Server info missing required field: {field}")
        
        logger.info(f"Executing tool call on server {server_info['url']}")
        
        result = await _execute_tool_call_simple(
            response=tool_call,
            server_path=server_info["url"],
            connection=server_info.get("connection"),
            output_type=output_type,
            transport=transport,
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
        _log_error_with_traceback(e, f"executing tool on server {server_info.get('url', 'unknown')}")
        
        if isinstance(e, (MCPValidationError, MCPExecutionError)):
            raise
        
        # Re-raise with appropriate MCP exception type
        if "validation" in str(e).lower() or "invalid" in str(e).lower():
            raise MCPValidationError(
                f"Invalid parameters for tool execution: {str(e)}",
                original_error=e,
                invalid_data={"tool_call": tool_call, "server_info": server_info}
            )
        else:
            raise MCPExecutionError(
                f"Failed to execute tool on server: {str(e)}",
                original_error=e,
                operation="execute_tool_on_server",
                context={"tool_call": tool_call, "server_info": server_info, "output_type": output_type, "transport": transport}
            )


async def execute_multiple_tools_on_multiple_mcp_servers(
    responses: List[Dict[str, Any]],
    urls: List[str],
    connections: List[MCPConnection] = None,
    output_type: Literal["json", "dict", "str", "formatted"] = "str",
    max_concurrent: Optional[int] = None,
    transport: str = "sse",
    *args,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Execute multiple tool calls across multiple MCP servers.
    Args:
        responses (List[Dict[str, Any]]): List of tool call requests.
        urls (List[str]): List of server URLs.
        connections (List[MCPConnection]): Optional list of MCPConnection objects.
        output_type (Literal): Output format type.
        max_concurrent (Optional[int]): Max concurrent tasks.
        transport (str): Transport type.
    Returns:
        List[Dict[str, Any]]: List of execution results.
    Raises:
        MCPValidationError: If parameters are invalid.
        MCPExecutionError: If tool execution fails.
    """
    try:
        if not responses:
            logger.warning("No responses provided for execution")
            return []
        
        if not urls:
            raise MCPValidationError("No server URLs provided")
        
        if not isinstance(responses, list):
            raise MCPValidationError(f"Responses must be a list, got {type(responses)}")
        
        if not isinstance(urls, list):
            raise MCPValidationError(f"URLs must be a list, got {type(urls)}")
        
        logger.info(
            f"Creating tool mapping for {len(urls)} servers using transport: {transport}"
        )
        
        server_tool_mapping = await _create_server_tool_mapping_async(
            urls=urls,
            connections=connections,
            format="openai",
            transport=transport,
        )
        
        if not server_tool_mapping:
            raise MCPExecutionError(
                "No tools found on any of the provided servers",
                operation="execute_multiple_tools_on_multiple_mcp_servers",
                context={"urls": urls, "connections": connections}
            )
        
        logger.info(
            f"Found {len(server_tool_mapping)} unique functions across all servers"
        )
        
        all_tool_calls = []
        logger.info(
            f"Processing {len(responses)} responses for tool call extraction"
        )
        
        # Handle character-by-character response reconstruction
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
                
                # Validate reconstructed JSON
                try:
                    json.loads(reconstructed_response)
                    logger.info(
                        "Successfully validated reconstructed JSON response"
                    )
                    responses = [reconstructed_response]
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
            except Exception as e:
                logger.warning(
                    f"Failed to reconstruct response from characters: {str(e)}"
                )
                _log_error_with_traceback(e, "reconstructing response from characters")
        
        # Process each response to extract tool calls
        for i, response in enumerate(responses):
            logger.debug(
                f"Processing response {i}: {type(response)} - {response}"
            )
            
            try:
                if isinstance(response, str):
                    try:
                        parsed_response = _safe_json_parse(response, f"response {i}")
                        if parsed_response is None:
                            logger.warning(f"Failed to parse JSON response at index {i}")
                            continue
                        response = parsed_response
                        logger.debug(f"Parsed JSON string response {i}: {response}")
                    except Exception as e:
                        logger.warning(f"Failed to parse response at index {i}: {str(e)}")
                        _log_error_with_traceback(e, f"parsing response {i}")
                        continue
                
                if isinstance(response, dict):
                    if "function" in response:
                        logger.debug(
                            f"Found single tool call in response {i}: {response['function']}"
                        )
                        
                        # Parse function arguments if they're a string
                        if isinstance(
                            response["function"].get("arguments"), str
                        ):
                            try:
                                parsed_args = _safe_json_parse(
                                    response["function"]["arguments"],
                                    f"function arguments in response {i}"
                                )
                                if parsed_args is not None:
                                    response["function"]["arguments"] = parsed_args
                                    logger.debug(
                                        f"Parsed function arguments: {response['function']['arguments']}"
                                    )
                                else:
                                    logger.warning(f"Failed to parse function arguments in response {i}")
                            except Exception as e:
                                logger.warning(f"Failed to parse function arguments in response {i}: {str(e)}")
                                _log_error_with_traceback(e, f"parsing function arguments in response {i}")
                        
                        all_tool_calls.append((i, response))
                        
                    elif "tool_calls" in response:
                        logger.debug(
                            f"Found multiple tool calls in response {i}: {len(response['tool_calls'])} calls"
                        )
                        
                        for tool_call in response["tool_calls"]:
                            # Parse tool call arguments if they're a string
                            if isinstance(
                                tool_call.get("function", {}).get("arguments"), str
                            ):
                                try:
                                    parsed_args = _safe_json_parse(
                                        tool_call["function"]["arguments"],
                                        f"tool call arguments in response {i}"
                                    )
                                    if parsed_args is not None:
                                        tool_call["function"]["arguments"] = parsed_args
                                        logger.debug(
                                            f"Parsed tool call arguments: {tool_call['function']['arguments']}"
                                        )
                                    else:
                                        logger.warning(f"Failed to parse tool call arguments in response {i}")
                                except Exception as e:
                                    logger.warning(f"Failed to parse tool call arguments in response {i}: {str(e)}")
                                    _log_error_with_traceback(e, f"parsing tool call arguments in response {i}")
                            
                            all_tool_calls.append((i, tool_call))
                            
                    elif "name" in response and "arguments" in response:
                        logger.debug(
                            f"Found direct tool call in response {i}: {response}"
                        )
                        
                        # Parse arguments if they're a string
                        if isinstance(response.get("arguments"), str):
                            try:
                                parsed_args = _safe_json_parse(
                                    response["arguments"],
                                    f"direct tool call arguments in response {i}"
                                )
                                if parsed_args is not None:
                                    response["arguments"] = parsed_args
                                    logger.debug(
                                        f"Parsed direct tool call arguments: {response['arguments']}"
                                    )
                                else:
                                    logger.warning(f"Failed to parse direct tool call arguments in response {i}")
                            except Exception as e:
                                logger.warning(f"Failed to parse direct tool call arguments in response {i}: {str(e)}")
                                _log_error_with_traceback(e, f"parsing direct tool call arguments in response {i}")
                        
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
                    
            except Exception as e:
                logger.error(f"Failed to process response {i}: {str(e)}")
                _log_error_with_traceback(e, f"processing response {i}")
                continue
        
        if not all_tool_calls:
            logger.warning("No tool calls found in responses")
            return []
        
        logger.info(f"Found {len(all_tool_calls)} tool calls to execute")
        max_concurrent = max_concurrent or len(all_tool_calls)
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_with_semaphore(tool_call_info):
            async with semaphore:
                try:
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
                        transport=transport,
                    )
                    result["response_index"] = response_index
                    return result
                    
                except Exception as e:
                    _log_error_with_traceback(e, "executing tool call with semaphore")
                    return {
                        "response_index": tool_call_info[0] if tool_call_info else -1,
                        "function_name": tool_call_info[1].get("function", {}).get("name", "unknown") if tool_call_info and len(tool_call_info) > 1 else "unknown",
                        "result": None,
                        "error": str(e),
                        "status": "exception",
                    }

        tasks = [
            execute_with_semaphore(tool_call_info)
            for tool_call_info in all_tool_calls
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        processed_results = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Task {i} failed with exception: {str(result)}"
                )
                _log_error_with_traceback(result, f"task {i} execution")
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
        
    except Exception as e:
        _log_error_with_traceback(e, "execute_multiple_tools_on_multiple_mcp_servers")
        
        if isinstance(e, (MCPValidationError, MCPExecutionError)):
            raise
        
        # Re-raise with appropriate MCP exception type
        if "validation" in str(e).lower() or "invalid" in str(e).lower():
            raise MCPValidationError(
                f"Invalid parameters for multiple tool execution: {str(e)}",
                original_error=e,
                invalid_data={"responses": responses, "urls": urls, "connections": connections}
            )
        else:
            raise MCPExecutionError(
                f"Failed to execute multiple tools: {str(e)}",
                original_error=e,
                operation="execute_multiple_tools_on_multiple_mcp_servers",
                context={"responses": responses, "urls": urls, "connections": connections, "output_type": output_type, "transport": transport}
            )


def execute_multiple_tools_on_multiple_mcp_servers_sync(
    responses: List[Dict[str, Any]],
    urls: List[str],
    connections: List[MCPConnection] = None,
    output_type: Literal["json", "dict", "str", "formatted"] = "str",
    max_concurrent: Optional[int] = None,
    transport: str = "sse",
    *args,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Synchronous version of execute_multiple_tools_on_multiple_mcp_servers.
    Args:
        responses (List[Dict[str, Any]]): List of tool call requests.
        urls (List[str]): List of server URLs.
        connections (List[MCPConnection]): Optional list of MCPConnection objects.
        output_type (Literal): Output format type.
        max_concurrent (Optional[int]): Max concurrent tasks.
        transport (str): Transport type.
    Returns:
        List[Dict[str, Any]]: List of execution results.
    Raises:
        MCPValidationError: If parameters are invalid.
        MCPExecutionError: If tool execution fails.
    """
    try:
        logger.info(
            f"execute_multiple_tools_on_multiple_mcp_servers_sync called for {len(urls)} urls"
        )
        
        if transport is None:
            transport = "sse"
            logger.info(f"Using default transport: {transport}")
        
        # Use the enhanced event loop runner with fallback
        return run_in_event_loop_with_fallback(
            execute_multiple_tools_on_multiple_mcp_servers(
                responses=responses,
                urls=urls,
                connections=connections,
                output_type=output_type,
                max_concurrent=max_concurrent,
                transport=transport,
                *args,
                **kwargs,
            )
        )
                
    except Exception as e:
        _log_error_with_traceback(e, "execute_multiple_tools_on_multiple_mcp_servers_sync")
        
        if isinstance(e, (MCPValidationError, MCPExecutionError)):
            raise
        
        # Re-raise with appropriate MCP exception type
        if "validation" in str(e).lower() or "invalid" in str(e).lower():
            raise MCPValidationError(
                f"Invalid parameters for multiple tool execution sync: {str(e)}",
                original_error=e,
                invalid_data={"responses": responses, "urls": urls, "connections": connections}
            )
        else:
            raise MCPExecutionError(
                f"Failed to execute multiple tools sync: {str(e)}",
                original_error=e,
                operation="execute_multiple_tools_on_multiple_mcp_servers_sync",
                context={"responses": responses, "urls": urls, "connections": connections, "output_type": output_type, "transport": transport}
            )
