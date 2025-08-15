import asyncio
import json
import logging
import time
import traceback
import re
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from functools import wraps

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession
from mcp.types import CallToolResult, TextContent

logger = logging.getLogger(__name__)

# MCP Exception classes
class MCPConnectionError(Exception):
    """Exception raised when there's an error connecting to the MCP server."""
    pass

class MCPExecutionError(Exception):
    """Exception raised when there's an error executing an MCP tool."""
    pass

class MCPToolError(Exception):
    """Exception raised when there's an error with a specific MCP tool."""
    pass

class MCPValidationError(Exception):
    """Exception raised when there's a validation error with MCP data."""
    pass

def retry_on_failure(max_retries: int = 3, base_delay: float = 1.0):
    """Retry decorator for MCP operations."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Failed after {max_retries} retries: {str(e)}")
                        raise last_exception
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def auto_detect_transport(url: str) -> str:
    """Auto-detect transport type from URL."""
    if url.startswith("stdio://"):
        return "stdio"
    elif url.startswith("http://") or url.startswith("https://"):
        return "http"
    else:
        # Default to stdio for file paths
        return "stdio"

def get_mcp_client(transport: str, url: str):
    """Get MCP client based on transport type."""
    logger.info(f"Getting MCP client for transport '{transport}' and url '{url}'.")
    
    if transport == "stdio":
        # Extract the command from stdio URL
        if url.startswith("stdio://"):
            command_path = url[8:]  # Remove "stdio://" prefix
            command_parts = command_path.split()
            command = command_parts[0]
            args = command_parts[1:] if len(command_parts) > 1 else []
            
            # Use the current Python executable for Windows compatibility
            import sys
            python_executable = sys.executable
            
            logger.info(f"Using stdio server parameters: command='{python_executable}' args={[command] + args}")
            
            # Use the correct API for MCP 1.11.0 with StdioServerParameters
            server_params = StdioServerParameters(
                command=python_executable,
                args=[command] + args
            )
            
            return stdio_client(server_params)
        else:
            raise ValueError(f"Invalid stdio URL format: {url}")
    
    elif transport == "http":
        return streamablehttp_client(url)
    
    else:
        raise ValueError(f"Unsupported transport type: {transport}")

@retry_on_failure(max_retries=3, base_delay=1.0)
async def aget_mcp_tools(
    server_path: str,
    transport: Optional[str] = None,
    *args,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Async function to get MCP tools from a server.
    
    Args:
        server_path: The server URL or path
        transport: The transport type (auto-detected if None)
        *args: Additional arguments
        **kwargs: Additional keyword arguments
        
    Returns:
        List of MCP tools
        
    Raises:
        MCPConnectionError: If connection fails
        MCPToolError: If tool retrieval fails
    """
    logger.info(f"aget_mcp_tools called for server_path: {server_path}")
    
    # Auto-detect transport if not specified
    if transport is None:
        transport = auto_detect_transport(server_path)
    
    logger.info(f"Fetching MCP tools from server: {server_path} using transport: {transport}")
    
    try:
        # Get the appropriate client
        logger.info(f"Getting MCP client for transport '{transport}' and url '{server_path}'.")
        client = get_mcp_client(transport, server_path)
        
        # Use the client as a context manager
        async with client as (read_stream, write_stream):
            # Create a session manually with the streams
            session = ClientSession(read_stream, write_stream)
            
            # Initialize the session without any parameters
            await session.initialize()
            
            # Get the tools
            tools = await session.list_tools()
            
            logger.info(f"Successfully retrieved {len(tools)} MCP tools")
            return tools
            
    except Exception as e:
        logger.error(f"Error fetching MCP tools: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        raise

def get_mcp_tools_sync(
    server_path: str,
    transport: Optional[str] = None,
    *args,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for aget_mcp_tools.
    
    Args:
        server_path: The server URL or path
        transport: The transport type (auto-detected if None)
        *args: Additional arguments
        **kwargs: Additional keyword arguments
        
    Returns:
        List of MCP tools
    """
    logger.info(f"get_mcp_tools_sync called for server_path: {server_path}")
    
    try:
        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an async context, we need to handle this differently
            logger.warning("Running in async context, creating new event loop")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        except RuntimeError:
            # No running loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            aget_mcp_tools(
                server_path=server_path,
                transport=transport,
                *args,
                **kwargs,
            )
        )
    except Exception as e:
        logger.error(f"Error in get_mcp_tools_sync: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

async def execute_tool_call_simple(
    server_path: str,
    tool_name: str,
    arguments: Dict[str, Any],
    transport: Optional[str] = None,
) -> str:
    """
    Execute a simple tool call and return the result as a string.
    
    Args:
        server_path: The server URL or path
        tool_name: Name of the tool to call
        arguments: Arguments for the tool
        transport: The transport type (auto-detected if None)
        
    Returns:
        Tool result as a string
    """
    logger.info(f"execute_tool_call_simple called for server_path: {server_path}")
    
    # Auto-detect transport if not specified
    if transport is None:
        transport = auto_detect_transport(server_path)
    
    try:
        # Get the appropriate client
        client = get_mcp_client(transport, server_path)
        
        # Use the client as a context manager
        async with client as (read_stream, write_stream):
            # Create a session manually with the streams
            session = ClientSession(read_stream, write_stream)
            
            # Initialize the session
            await session.initialize()
            
            # Call the tool
            result = await session.call_tool(tool_name, arguments)
            
            # Convert result to string
            if result and hasattr(result, 'content') and result.content:
                # Extract text content from the result
                text_content = ""
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        text_content += content_item.text
                return text_content
            else:
                return str(result) if result else ""
                
    except Exception as e:
        logger.error(f"Error executing tool call: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return f"Error executing tool {tool_name}: {str(e)}"

def execute_tool_call_simple_sync(
    server_path: str,
    tool_name: str,
    arguments: Dict[str, Any],
    transport: Optional[str] = None,
) -> str:
    """
    Synchronous wrapper for execute_tool_call_simple.
    
    Args:
        server_path: The server URL or path
        tool_name: Name of the tool to call
        arguments: Arguments for the tool
        transport: The transport type (auto-detected if None)
        
    Returns:
        Tool result as a string
    """
    logger.info(f"execute_tool_call_simple_sync called for server_path: {server_path}")
    
    try:
        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an async context, we need to handle this differently
            logger.warning("Running in async context, creating new event loop")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        except RuntimeError:
            # No running loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            execute_tool_call_simple(
                server_path=server_path,
                tool_name=tool_name,
                arguments=arguments,
                transport=transport,
            )
        )
    except Exception as e:
        logger.error(f"Error in execute_tool_call_simple_sync: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return f"Error executing tool {tool_name}: {str(e)}"

# Advanced functionality - Tool call extraction and parsing
def _extract_tool_calls_from_response(response: str) -> List[Dict[str, Any]]:
    """
    Extract tool calls from LLM response with advanced parsing capabilities.
    
    Args:
        response: The response string from the LLM
        
    Returns:
        List of tool call dictionaries
    """
    tool_calls = []
    
    try:
        # Try to find JSON tool calls in code blocks
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                tool_data = json.loads(json_match.group(1))
                
                # Check for tool_uses format (OpenAI format)
                if "tool_uses" in tool_data and tool_data["tool_uses"]:
                    for tool_call in tool_data["tool_uses"]:
                        if "recipient_name" in tool_call:
                            tool_name = tool_call["recipient_name"]
                            arguments = tool_call.get("parameters", {})
                            tool_calls.append({
                                "name": tool_name,
                                "arguments": arguments
                            })
                
                # Check for direct tool call format
                elif "name" in tool_data and "arguments" in tool_data:
                    tool_calls.append({
                        "name": tool_data["name"],
                        "arguments": tool_data["arguments"]
                    })
                
                # Check for function_calls format
                elif "function_calls" in tool_data and tool_data["function_calls"]:
                    for tool_call in tool_data["function_calls"]:
                        if "name" in tool_call and "arguments" in tool_call:
                            tool_calls.append({
                                "name": tool_call["name"],
                                "arguments": tool_call["arguments"]
                            })
                
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON tool calls without code blocks
        if not tool_calls:
            json_patterns = [
                r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\}',
                r'\{[^{}]*"tool_uses"[^{}]*\}',
                r'\{[^{}]*"function_calls"[^{}]*\}'
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, response, re.DOTALL)
                for match in matches:
                    try:
                        tool_data = json.loads(match)
                        
                        # Check for tool_uses format
                        if "tool_uses" in tool_data and tool_data["tool_uses"]:
                            for tool_call in tool_data["tool_uses"]:
                                if "recipient_name" in tool_call:
                                    tool_calls.append({
                                        "name": tool_call["recipient_name"],
                                        "arguments": tool_call.get("parameters", {})
                                    })
                        
                        # Check for direct tool call format
                        elif "name" in tool_data and "arguments" in tool_data:
                            tool_calls.append({
                                "name": tool_data["name"],
                                "arguments": tool_data["arguments"]
                            })
                        
                        # Check for function_calls format
                        elif "function_calls" in tool_data and tool_data["function_calls"]:
                            for tool_call in tool_data["function_calls"]:
                                if "name" in tool_call and "arguments" in tool_call:
                                    tool_calls.append({
                                        "name": tool_call["name"],
                                        "arguments": tool_call["arguments"]
                                    })
                    
                    except json.JSONDecodeError:
                        continue
        
        # If no JSON found, try to extract from text using pattern matching
        if not tool_calls:
            response_lower = response.lower()
            
            # Look for mathematical expressions
            if "calculate" in response_lower or "compute" in response_lower or "math" in response_lower:
                # Extract mathematical expression
                expr_patterns = [
                    r'(\d+\s*[\+\-\*\/\^]\s*\d+)',
                    r'calculate\s+(.+?)(?:\n|\.|$)',
                    r'compute\s+(.+?)(?:\n|\.|$)'
                ]
                
                for pattern in expr_patterns:
                    expr_match = re.search(pattern, response, re.IGNORECASE)
                    if expr_match:
                        expression = expr_match.group(1).strip()
                        tool_calls.append({
                            "name": "calculate",
                            "arguments": {"expression": expression}
                        })
                        break
                
                # Default calculation if no expression found
                if not any("calculate" in tc.get("name", "") for tc in tool_calls):
                    tool_calls.append({
                        "name": "calculate",
                        "arguments": {"expression": "2+2"}
                    })
            
            # Look for search operations
            elif "search" in response_lower or "find" in response_lower or "look up" in response_lower:
                # Extract search query
                search_patterns = [
                    r'search\s+for\s+(.+?)(?:\n|\.|$)',
                    r'find\s+(.+?)(?:\n|\.|$)',
                    r'look up\s+(.+?)(?:\n|\.|$)'
                ]
                
                for pattern in search_patterns:
                    search_match = re.search(pattern, response, re.IGNORECASE)
                    if search_match:
                        query = search_match.group(1).strip()
                        tool_calls.append({
                            "name": "search",
                            "arguments": {"query": query}
                        })
                        break
                
                # Default search if no query found
                if not any("search" in tc.get("name", "") for tc in tool_calls):
                    tool_calls.append({
                        "name": "search",
                        "arguments": {"query": response.strip()}
                    })
            
            # Look for file operations
            elif "read" in response_lower or "file" in response_lower or "open" in response_lower:
                # Extract file path
                file_patterns = [
                    r'read\s+(.+?)(?:\n|\.|$)',
                    r'open\s+(.+?)(?:\n|\.|$)',
                    r'file\s+(.+?)(?:\n|\.|$)'
                ]
                
                for pattern in file_patterns:
                    file_match = re.search(pattern, response, re.IGNORECASE)
                    if file_match:
                        file_path = file_match.group(1).strip()
                        tool_calls.append({
                            "name": "read_file",
                            "arguments": {"file_path": file_path}
                        })
                        break
            
            # Look for web operations
            elif "web" in response_lower or "url" in response_lower or "http" in response_lower:
                # Extract URL
                url_patterns = [
                    r'https?://[^\s]+',
                    r'www\.[^\s]+',
                    r'url\s+(.+?)(?:\n|\.|$)'
                ]
                
                for pattern in url_patterns:
                    url_match = re.search(pattern, response, re.IGNORECASE)
                    if url_match:
                        url = url_match.group(0) if pattern.startswith('http') else url_match.group(1).strip()
                        tool_calls.append({
                            "name": "fetch_url",
                            "arguments": {"url": url}
                        })
                        break
            
            # Default tool call if no specific patterns found
            else:
                tool_calls.append({
                    "name": "default_tool",
                    "arguments": {"input": response.strip()}
                })
        
    except Exception as e:
        logger.error(f"Error extracting tool calls: {e}")
        # Return default tool call
        tool_calls.append({
            "name": "default_tool",
            "arguments": {"input": response.strip()}
        })
    
    return tool_calls

# Advanced function for handling complex responses with multiple tool calls
async def execute_tool_calls_from_response(
    response: Any,
    server_path: str,
    transport: Optional[str] = None,
    max_concurrent: int = 3
) -> List[Dict[str, Any]]:
    """
    Execute multiple tool calls extracted from an LLM response.
    
    Args:
        response: The response from the LLM (may contain tool calls)
        server_path: MCP server path/URL
        transport: Transport type (auto-detected if None)
        max_concurrent: Maximum concurrent tool executions
        
    Returns:
        List of tool execution results
    """
    try:
        # Extract tool calls from response
        if isinstance(response, str):
            tool_calls = _extract_tool_calls_from_response(response)
        elif hasattr(response, 'choices') and response.choices:
            # Handle OpenAI-style response objects
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'tool_calls'):
                tool_calls = []
                for tool_call in choice.message.tool_calls:
                    tool_calls.append({
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments)
                    })
            else:
                tool_calls = _extract_tool_calls_from_response(str(response))
        else:
            tool_calls = [{"name": "default_tool", "arguments": {}}]
        
        # Execute tool calls
        results = []
        
        if max_concurrent > 1 and len(tool_calls) > 1:
            # Execute concurrently
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def execute_single_tool(tool_call):
                async with semaphore:
                    try:
                        result = await execute_tool_call_simple(
                            server_path=server_path,
                            tool_name=tool_call["name"],
                            arguments=tool_call["arguments"],
                            transport=transport
                        )
                        return {
                            "success": True,
                            "tool_name": tool_call["name"],
                            "arguments": tool_call["arguments"],
                            "result": result
                        }
                    except Exception as e:
                        logger.error(f"Error executing tool {tool_call['name']}: {e}")
                        return {
                            "success": False,
                            "tool_name": tool_call["name"],
                            "arguments": tool_call["arguments"],
                            "error": str(e)
                        }
            
            # Execute all tools concurrently
            tasks = [execute_single_tool(tool_call) for tool_call in tool_calls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            final_results = []
            for result in results:
                if isinstance(result, Exception):
                    final_results.append({
                        "success": False,
                        "error": str(result)
                    })
                else:
                    final_results.append(result)
            
            results = final_results
        
        else:
            # Execute sequentially
            for tool_call in tool_calls:
                try:
                    result = await execute_tool_call_simple(
                        server_path=server_path,
                        tool_name=tool_call["name"],
                        arguments=tool_call["arguments"],
                        transport=transport
                    )
                    results.append({
                        "success": True,
                        "tool_name": tool_call["name"],
                        "arguments": tool_call["arguments"],
                        "result": result
                    })
                except Exception as e:
                    logger.error(f"Error executing tool {tool_call['name']}: {e}")
                    results.append({
                        "success": False,
                        "tool_name": tool_call["name"],
                        "arguments": tool_call["arguments"],
                        "error": str(e)
                    })
        
        return results
        
    except Exception as e:
        logger.error(f"Error in execute_tool_calls_from_response: {e}")
        return [{"success": False, "error": str(e)}]

def execute_tool_calls_from_response_sync(
    response: Any,
    server_path: str,
    transport: Optional[str] = None,
    max_concurrent: int = 3
) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for execute_tool_calls_from_response.
    
    Args:
        response: The response from the LLM (may contain tool calls)
        server_path: MCP server path/URL
        transport: Transport type (auto-detected if None)
        max_concurrent: Maximum concurrent tool executions
        
    Returns:
        List of tool execution results
    """
    try:
        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an async context, we need to handle this differently
            logger.warning("Running in async context, creating new event loop")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        except RuntimeError:
            # No running loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            execute_tool_calls_from_response(
                response=response,
                server_path=server_path,
                transport=transport,
                max_concurrent=max_concurrent
            )
        )
    except Exception as e:
        logger.error(f"Error in execute_tool_calls_from_response_sync: {e}")
        return [{"success": False, "error": str(e)}]

# Advanced streaming functionality
async def execute_tool_call_streaming(
    server_path: str,
    tool_name: str,
    arguments: Dict[str, Any],
    transport: Optional[str] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Execute a tool call with streaming support.
    
    Args:
        server_path: The server URL or path
        tool_name: Name of the tool to call
        arguments: Arguments for the tool
        transport: The transport type (auto-detected if None)
        
    Yields:
        Streaming tool execution results
    """
    logger.info(f"execute_tool_call_streaming called for server_path: {server_path}")
    
    # Auto-detect transport if not specified
    if transport is None:
        transport = auto_detect_transport(server_path)
    
    try:
        # Get the appropriate client
        client = get_mcp_client(transport, server_path)
        
        # Use the client as a context manager
        async with client as (read_stream, write_stream):
            # Create a session manually with the streams
            session = ClientSession(read_stream, write_stream)
            
            # Initialize the session
            await session.initialize()
            
            # Check if streaming method exists
            if hasattr(session, 'call_tool_streaming'):
                # Use streaming method if available
                async for result in session.call_tool_streaming(tool_name, arguments):
                    yield {
                        "success": True,
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "result": result.model_dump() if hasattr(result, 'model_dump') else str(result),
                        "streaming": True
                    }
            else:
                # Fallback to non-streaming
                logger.warning("Streaming not available, falling back to non-streaming")
                result = await session.call_tool(tool_name, arguments)
                yield {
                    "success": True,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": result.model_dump() if hasattr(result, 'model_dump') else str(result),
                    "streaming": False
                }
                
    except Exception as e:
        logger.error(f"Error executing streaming tool call: {e}")
        yield {
            "success": False,
            "tool_name": tool_name,
            "arguments": arguments,
            "error": str(e),
            "streaming": False
        }

def execute_tool_call_streaming_sync(
    server_path: str,
    tool_name: str,
    arguments: Dict[str, Any],
    transport: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for execute_tool_call_streaming.
    
    Args:
        server_path: The server URL or path
        tool_name: Name of the tool to call
        arguments: Arguments for the tool
        transport: The transport type (auto-detected if None)
        
    Returns:
        List of streaming tool execution results
    """
    logger.info(f"execute_tool_call_streaming_sync called for server_path: {server_path}")
    
    try:
        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an async context, we need to handle this differently
            logger.warning("Running in async context, creating new event loop")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        except RuntimeError:
            # No running loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        results = []
        
        async def collect_streaming_results():
            async for result in execute_tool_call_streaming(
                server_path=server_path,
                tool_name=tool_name,
                arguments=arguments,
                transport=transport
            ):
                results.append(result)
        
        loop.run_until_complete(collect_streaming_results())
        return results
        
    except Exception as e:
        logger.error(f"Error in execute_tool_call_streaming_sync: {e}")
        return [{"success": False, "error": str(e)}]

# Advanced multiple server functionality
async def get_tools_for_multiple_mcp_servers(server_paths: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get tools from multiple MCP servers concurrently.
    
    Args:
        server_paths: List of server URLs or paths
        
    Returns:
        Dictionary mapping server paths to their tools
    """
    logger.info(f"Getting tools from {len(server_paths)} MCP servers")
    
    async def get_tools_for_single_server(server_path: str) -> tuple:
        try:
            tools = await aget_mcp_tools(server_path)
            return server_path, tools
        except Exception as e:
            logger.error(f"Error getting tools from {server_path}: {e}")
            return server_path, []
    
    # Execute concurrently
    tasks = [get_tools_for_single_server(server_path) for server_path in server_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    server_tools = {}
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Exception in get_tools_for_multiple_mcp_servers: {result}")
        else:
            server_path, tools = result
            server_tools[server_path] = tools
    
    return server_tools

def get_tools_for_multiple_mcp_servers_sync(server_paths: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Synchronous wrapper for get_tools_for_multiple_mcp_servers.
    
    Args:
        server_paths: List of server URLs or paths
        
    Returns:
        Dictionary mapping server paths to their tools
    """
    try:
        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an async context, we need to handle this differently
            logger.warning("Running in async context, creating new event loop")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        except RuntimeError:
            # No running loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            get_tools_for_multiple_mcp_servers(server_paths)
        )
    except Exception as e:
        logger.error(f"Error in get_tools_for_multiple_mcp_servers_sync: {e}")
        return {}

async def execute_multiple_tools_on_multiple_mcp_servers(
    server_tool_mappings: Dict[str, List[str]], 
    tool_arguments: Dict[str, Dict[str, Any]]
) -> Dict[str, str]:
    """
    Execute multiple tools on multiple servers concurrently.
    
    Args:
        server_tool_mappings: Dictionary mapping server paths to lists of tool names
        tool_arguments: Dictionary mapping tool names to their arguments
        
    Returns:
        Dictionary mapping tool names to their results
    """
    logger.info(f"Executing multiple tools on multiple servers")
    
    async def execute_tool_on_server(server_path: str, tool_name: str) -> tuple:
        try:
            arguments = tool_arguments.get(tool_name, {})
            result = await execute_tool_call_simple(server_path, tool_name, arguments)
            return tool_name, result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name} on {server_path}: {e}")
            return tool_name, f"Error: {str(e)}"
    
    # Create tasks for all tool executions
    tasks = []
    for server_path, tool_names in server_tool_mappings.items():
        for tool_name in tool_names:
            if tool_name in tool_arguments:
                tasks.append(execute_tool_on_server(server_path, tool_name))
    
    # Execute concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    tool_results = {}
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Exception in execute_multiple_tools_on_multiple_mcp_servers: {result}")
        else:
            tool_name, result_value = result
            tool_results[tool_name] = result_value
    
    return tool_results

def execute_multiple_tools_on_multiple_mcp_servers_sync(
    server_tool_mappings: Dict[str, List[str]], 
    tool_arguments: Dict[str, Dict[str, Any]]
) -> Dict[str, str]:
    """
    Synchronous wrapper for execute_multiple_tools_on_multiple_mcp_servers.
    
    Args:
        server_tool_mappings: Dictionary mapping server paths to lists of tool names
        tool_arguments: Dictionary mapping tool names to their arguments
        
    Returns:
        Dictionary mapping tool names to their results
    """
    try:
        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an async context, we need to handle this differently
            logger.warning("Running in async context, creating new event loop")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        except RuntimeError:
            # No running loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            execute_multiple_tools_on_multiple_mcp_servers(
                server_tool_mappings=server_tool_mappings,
                tool_arguments=tool_arguments
            )
        )
    except Exception as e:
        logger.error(f"Error in execute_multiple_tools_on_multiple_mcp_servers_sync: {e}")
        return {}

# Compatibility functions for backward compatibility
def _create_server_tool_mapping(server_path: str) -> Dict[str, Any]:
    """Create a mapping of tools for a server (placeholder)."""
    logger.warning("_create_server_tool_mapping is deprecated")
    return {}

async def _create_server_tool_mapping_async(server_path: str) -> Dict[str, Any]:
    """Create a mapping of tools for a server asynchronously (placeholder)."""
    logger.warning("_create_server_tool_mapping_async is deprecated")
    return {}

def _execute_tool_call_simple(server_path: str, tool_name: str, arguments: Dict[str, Any]) -> str:
    """Execute a tool call (synchronous wrapper)."""
    return execute_tool_call_simple_sync(server_path, tool_name, arguments)

async def _execute_tool_on_server(server_path: str, tool_name: str, arguments: Dict[str, Any]) -> str:
    """Execute a tool on a server (asynchronous)."""
    return await execute_tool_call_simple(server_path, tool_name, arguments)

# Compatibility function for the agent's response parameter
async def execute_tool_call_simple_with_response(response: Any, server_path: str) -> str:
    """
    Compatibility function that handles the response parameter from the agent.
    
    Args:
        response: The response from the LLM (contains tool call info)
        server_path: The server URL or path
        
    Returns:
        Tool result as a string
    """
    try:
        # Extract tool name and arguments from the response
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'tool_calls'):
                tool_calls = choice.message.tool_calls
                if tool_calls:
                    tool_call = tool_calls[0]
                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    
                    return await execute_tool_call_simple(server_path, tool_name, arguments)
        
        # Fallback: try to parse as JSON if it's a string
        if isinstance(response, str):
            try:
                data = json.loads(response)
                if 'tool_name' in data and 'arguments' in data:
                    return await execute_tool_call_simple(server_path, data['tool_name'], data['arguments'])
            except json.JSONDecodeError:
                pass
        
        # If we can't extract tool info, return an error message
        return f"Error: Could not extract tool information from response: {type(response)}"
        
    except Exception as e:
        logger.error(f"Error in execute_tool_call_simple_with_response: {e}")
        return f"Error executing tool: {str(e)}"

def get_or_create_event_loop():
    """
    Get the current event loop or create a new one if none exists.
    
    Returns:
        The event loop context manager
    """
    try:
        loop = asyncio.get_running_loop()
        # If we're already in an event loop, return a context manager that does nothing
        class NoOpContextManager:
            def __enter__(self):
                return loop
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return NoOpContextManager()
    except RuntimeError:
        # No running loop, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        class LoopContextManager:
            def __init__(self, loop):
                self.loop = loop
            def __enter__(self):
                return self.loop
            def __exit__(self, exc_type, exc_val, exc_tb):
                try:
                    self.loop.close()
                except:
                    pass
        
        return LoopContextManager(loop)
