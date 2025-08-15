"""
Unified MCP Client for Swarms Framework

This module provides a unified interface for MCP (Model Context Protocol) operations
with support for multiple transport types: stdio, http, streamable_http, and sse.

All transport types are optional and can be configured based on requirements.
Streaming support is included for real-time communication.

Dependencies:
- Core MCP: pip install mcp
- Streamable HTTP: pip install mcp[streamable-http] 
- HTTP transport: pip install httpx
- All dependencies are optional and gracefully handled

Transport Types:
- stdio: Local command-line tools (no additional deps)
- http: Standard HTTP communication (requires httpx)
- streamable_http: Real-time HTTP streaming (requires mcp[streamable-http])
- sse: Server-Sent Events (included with core mcp)
- auto: Auto-detection based on URL scheme
"""

import asyncio
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any, Dict, List, Literal, Optional, Union, AsyncGenerator, Callable
from urllib.parse import urlparse

from loguru import logger
from pydantic import BaseModel, Field

# Import existing MCP functionality
from swarms.schemas.mcp_schemas import MCPConnection
from swarms.tools.mcp_client_call import (
    MCPConnectionError,
    MCPExecutionError,
    MCPToolError,
    MCPValidationError,
    aget_mcp_tools,
    execute_multiple_tools_on_multiple_mcp_servers,
    execute_multiple_tools_on_multiple_mcp_servers_sync,
    execute_tool_call_simple,
    get_mcp_tools_sync,
    get_or_create_event_loop,
)

# Try to import MCP libraries
try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    logger.warning("MCP client libraries not available. Install with: pip install mcp")
    MCP_AVAILABLE = False

try:
    from mcp.client.streamable_http import streamablehttp_client
    STREAMABLE_HTTP_AVAILABLE = True
except ImportError:
    logger.warning("Streamable HTTP client not available. Install with: pip install mcp[streamable-http]")
    STREAMABLE_HTTP_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    logger.warning("HTTPX not available. Install with: pip install httpx")
    HTTPX_AVAILABLE = False


class UnifiedTransportConfig(BaseModel):
    """
    Unified configuration for MCP transport types.
    
    This extends the existing MCPConnection schema with additional
    transport-specific options and auto-detection capabilities.
    Includes streaming support for real-time communication.
    """
    
    # Transport type - can be auto-detected
    transport_type: Literal["stdio", "http", "streamable_http", "sse", "auto"] = Field(
        default="auto",
        description="The transport type to use. 'auto' enables auto-detection."
    )
    
    # Connection details
    url: Optional[str] = Field(
        default=None,
        description="URL for HTTP-based transports or stdio command path"
    )
    
    # STDIO specific
    command: Optional[List[str]] = Field(
        default=None,
        description="Command and arguments for stdio transport"
    )
    
    # HTTP specific
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="HTTP headers for HTTP-based transports"
    )
    
    # Common settings
    timeout: int = Field(
        default=30,
        description="Timeout in seconds"
    )
    
    authorization_token: Optional[str] = Field(
        default=None,
        description="Authentication token for accessing the MCP server"
    )
    
    # Auto-detection settings
    auto_detect: bool = Field(
        default=True,
        description="Whether to auto-detect transport type from URL"
    )
    
    # Fallback settings
    fallback_transport: Literal["stdio", "http", "streamable_http", "sse"] = Field(
        default="sse",
        description="Fallback transport if auto-detection fails"
    )
    
    # Streaming settings
    enable_streaming: bool = Field(
        default=True,
        description="Whether to enable streaming support"
    )
    
    streaming_timeout: Optional[int] = Field(
        default=None,
        description="Timeout for streaming operations"
    )
    
    streaming_callback: Optional[Callable[[str], None]] = Field(
        default=None,
        description="Optional callback function for streaming chunks"
    )


class MCPUnifiedClient:
    """
    Unified MCP client that supports multiple transport types.
    
    This client integrates with the existing swarms framework and provides
    a unified interface for all MCP operations with streaming support.
    """
    
    def __init__(self, config: Union[UnifiedTransportConfig, MCPConnection, str]):
        """
        Initialize the unified MCP client.
        
        Args:
            config: Transport configuration (UnifiedTransportConfig, MCPConnection, or URL string)
        """
        self.config = self._normalize_config(config)
        self._validate_config()
        
    def _normalize_config(self, config: Union[UnifiedTransportConfig, MCPConnection, str]) -> UnifiedTransportConfig:
        """
        Normalize different config types to UnifiedTransportConfig.
        
        Args:
            config: Configuration in various formats
            
        Returns:
            Normalized UnifiedTransportConfig
        """
        if isinstance(config, str):
            # URL string - create config with auto-detection
            return UnifiedTransportConfig(
                url=config,
                transport_type="auto",
                auto_detect=True,
                enable_streaming=True
            )
        elif isinstance(config, MCPConnection):
            # Convert existing MCPConnection to UnifiedTransportConfig
            return UnifiedTransportConfig(
                transport_type=config.transport or "auto",
                url=config.url,
                headers=config.headers,
                timeout=config.timeout or 30,
                authorization_token=config.authorization_token,
                auto_detect=True,
                enable_streaming=True
            )
        elif isinstance(config, UnifiedTransportConfig):
            return config
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")
    
    def _validate_config(self) -> None:
        """Validate the transport configuration."""
        if not MCP_AVAILABLE:
            raise ImportError("MCP client libraries are required")
            
        if self.config.transport_type == "streamable_http" and not STREAMABLE_HTTP_AVAILABLE:
            raise ImportError("Streamable HTTP transport requires mcp[streamable-http]")
            
        if self.config.transport_type == "http" and not HTTPX_AVAILABLE:
            raise ImportError("HTTP transport requires httpx")
    
    def _auto_detect_transport(self, url: str) -> str:
        """
        Auto-detect transport type from URL.
        
        Args:
            url: The URL to analyze
            
        Returns:
            Detected transport type
        """
        if not url:
            return "stdio"
            
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        
        if scheme in ("http", "https"):
            if STREAMABLE_HTTP_AVAILABLE and self.config.enable_streaming:
                return "streamable_http"
            else:
                return "http"
        elif scheme in ("ws", "wss"):
            return "sse"
        elif scheme == "" or "stdio" in url:
            return "stdio"
        else:
            return self.config.fallback_transport
    
    def _get_effective_transport(self) -> str:
        """
        Get the effective transport type after auto-detection.
        
        Returns:
            Effective transport type
        """
        transport = self.config.transport_type
        
        if transport == "auto" and self.config.auto_detect and self.config.url:
            transport = self._auto_detect_transport(self.config.url)
            logger.info(f"Auto-detected transport type: {transport}")
        
        return transport
    
    @asynccontextmanager
    async def get_client_context(self):
        """
        Get the appropriate MCP client context manager.
        
        Yields:
            MCP client context manager
        """
        transport_type = self._get_effective_transport()
        
        if transport_type == "stdio":
            command = self.config.command or [self.config.url] if self.config.url else None
            if not command:
                raise ValueError("Command is required for stdio transport")
            async with stdio_client(command) as (read, write):
                yield read, write
                
        elif transport_type == "streamable_http":
            if not STREAMABLE_HTTP_AVAILABLE:
                raise ImportError("Streamable HTTP transport not available")
            if not self.config.url:
                raise ValueError("URL is required for streamable_http transport")
            async with streamablehttp_client(
                self.config.url,
                headers=self.config.headers,
                timeout=self.config.streaming_timeout or self.config.timeout
            ) as (read, write):
                yield read, write
                
        elif transport_type == "http":
            if not HTTPX_AVAILABLE:
                raise ImportError("HTTP transport requires httpx")
            if not self.config.url:
                raise ValueError("URL is required for http transport")
            async with self._http_client_context() as (read, write):
                yield read, write
                
        elif transport_type == "sse":
            if not self.config.url:
                raise ValueError("URL is required for sse transport")
            async with sse_client(
                self.config.url,
                headers=self.config.headers,
                timeout=self.config.streaming_timeout or self.config.timeout
            ) as (read, write):
                yield read, write
        else:
            raise ValueError(f"Unsupported transport type: {transport_type}")
    
    @asynccontextmanager
    async def _http_client_context(self):
        """
        HTTP client context manager using httpx.
        
        Yields:
            Tuple of (read, write) functions
        """
        if not HTTPX_AVAILABLE:
            raise ImportError("HTTPX is required for HTTP transport")
            
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            # Create read/write functions for HTTP transport
            async def read():
                # Implement HTTP read logic for MCP
                try:
                    response = await client.get(self.config.url)
                    response.raise_for_status()
                    return response.text
                except Exception as e:
                    logger.error(f"HTTP read error: {e}")
                    raise MCPConnectionError(f"HTTP read failed: {e}")
                
            async def write(data):
                # Implement HTTP write logic for MCP
                try:
                    response = await client.post(
                        self.config.url,
                        json=data,
                        headers=self.config.headers or {}
                    )
                    response.raise_for_status()
                    return response.json()
                except Exception as e:
                    logger.error(f"HTTP write error: {e}")
                    raise MCPConnectionError(f"HTTP write failed: {e}")
                
            yield read, write
    
    async def get_tools(self, format: Literal["mcp", "openai"] = "openai") -> List[Dict[str, Any]]:
        """
        Get available tools from the MCP server.
        
        Args:
            format: Output format for tools
            
        Returns:
            List of available tools
        """
        async with self.get_client_context() as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                
                if format == "openai":
                    return [self._convert_mcp_tool_to_openai(tool) for tool in tools.tools]
                else:
                    return [tool.model_dump() for tool in tools.tools]
    
    def _convert_mcp_tool_to_openai(self, mcp_tool) -> Dict[str, Any]:
        """
        Convert MCP tool to OpenAI format.
        
        Args:
            mcp_tool: MCP tool object
            
        Returns:
            OpenAI-compatible tool format
        """
        return {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": mcp_tool.description or "",
                "parameters": mcp_tool.inputSchema
            }
        }
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        async with self.get_client_context() as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name=tool_name, arguments=arguments)
                return result.model_dump()
    
    async def call_tool_streaming(self, tool_name: str, arguments: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Call a tool on the MCP server with streaming support.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Yields:
            Streaming tool execution results
        """
        if not self.config.enable_streaming:
            # Fallback to non-streaming
            result = await self.call_tool(tool_name, arguments)
            yield result
            return
            
        async with self.get_client_context() as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Use streaming call if available
                try:
                    # Check if streaming method exists
                    if hasattr(session, 'call_tool_streaming'):
                        async for result in session.call_tool_streaming(name=tool_name, arguments=arguments):
                            yield result.model_dump()
                    else:
                        # Fallback to non-streaming if streaming not available
                        logger.warning("Streaming not available in MCP session, falling back to non-streaming")
                        result = await session.call_tool(name=tool_name, arguments=arguments)
                        yield result.model_dump()
                except AttributeError:
                    # Fallback to non-streaming if streaming not available
                    logger.warning("Streaming method not found, falling back to non-streaming")
                    result = await session.call_tool(name=tool_name, arguments=arguments)
                    yield result.model_dump()
                except Exception as e:
                    logger.error(f"Error in streaming tool call: {e}")
                    # Final fallback to non-streaming
                    try:
                        result = await session.call_tool(name=tool_name, arguments=arguments)
                        yield result.model_dump()
                    except Exception as fallback_error:
                        logger.error(f"Fallback tool call also failed: {fallback_error}")
                        raise MCPExecutionError(f"Tool call failed: {fallback_error}")
    
    def get_tools_sync(self, format: Literal["mcp", "openai"] = "openai") -> List[Dict[str, Any]]:
        """
        Synchronous version of get_tools.
        
        Args:
            format: Output format for tools
            
        Returns:
            List of available tools
        """
        with get_or_create_event_loop() as loop:
            try:
                return loop.run_until_complete(self.get_tools(format=format))
            except Exception as e:
                logger.error(f"Error in get_tools_sync: {str(e)}")
                raise MCPExecutionError(f"Failed to get tools sync: {str(e)}")
    
    def call_tool_sync(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous version of call_tool.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        with get_or_create_event_loop() as loop:
            try:
                return loop.run_until_complete(self.call_tool(tool_name, arguments))
            except Exception as e:
                logger.error(f"Error in call_tool_sync: {str(e)}")
                raise MCPExecutionError(f"Failed to call tool sync: {str(e)}")
    
    def call_tool_streaming_sync(self, tool_name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Synchronous version of call_tool_streaming.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            List of streaming tool execution results
        """
        with get_or_create_event_loop() as loop:
            try:
                results = []
                async def collect_streaming_results():
                    async for result in self.call_tool_streaming(tool_name, arguments):
                        results.append(result)
                loop.run_until_complete(collect_streaming_results())
                return results
            except Exception as e:
                logger.error(f"Error in call_tool_streaming_sync: {str(e)}")
                raise MCPExecutionError(f"Failed to call tool streaming sync: {str(e)}")


# Enhanced functions that work with the unified client
def get_mcp_tools_unified(
    config: Union[UnifiedTransportConfig, MCPConnection, str],
    format: Literal["mcp", "openai"] = "openai"
) -> List[Dict[str, Any]]:
    """
    Get MCP tools using the unified client.
    
    Args:
        config: Transport configuration
        format: Output format for tools
        
    Returns:
        List of available tools
    """
    client = MCPUnifiedClient(config)
    return client.get_tools_sync(format=format)


async def aget_mcp_tools_unified(
    config: Union[UnifiedTransportConfig, MCPConnection, str],
    format: Literal["mcp", "openai"] = "openai"
) -> List[Dict[str, Any]]:
    """
    Async version of get_mcp_tools_unified.
    
    Args:
        config: Transport configuration
        format: Output format for tools
        
    Returns:
        List of available tools
    """
    client = MCPUnifiedClient(config)
    return await client.get_tools(format=format)


def execute_tool_call_unified(
    config: Union[UnifiedTransportConfig, MCPConnection, str],
    tool_name: str,
    arguments: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute a tool call using the unified client.
    
    Args:
        config: Transport configuration
        tool_name: Name of the tool to call
        arguments: Tool arguments
        
    Returns:
        Tool execution result
    """
    client = MCPUnifiedClient(config)
    return client.call_tool_sync(tool_name, arguments)


async def aexecute_tool_call_unified(
    config: Union[UnifiedTransportConfig, MCPConnection, str],
    tool_name: str,
    arguments: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Async version of execute_tool_call_unified.
    
    Args:
        config: Transport configuration
        tool_name: Name of the tool to call
        arguments: Tool arguments
        
    Returns:
        Tool execution result
    """
    client = MCPUnifiedClient(config)
    return await client.call_tool(tool_name, arguments)


def execute_tool_call_streaming_unified(
    config: Union[UnifiedTransportConfig, MCPConnection, str],
    tool_name: str,
    arguments: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Execute a tool call with streaming using the unified client.
    
    Args:
        config: Transport configuration
        tool_name: Name of the tool to call
        arguments: Tool arguments
        
    Returns:
        List of streaming tool execution results
    """
    client = MCPUnifiedClient(config)
    return client.call_tool_streaming_sync(tool_name, arguments)


async def aexecute_tool_call_streaming_unified(
    config: Union[UnifiedTransportConfig, MCPConnection, str],
    tool_name: str,
    arguments: Dict[str, Any]
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Async version of execute_tool_call_streaming_unified.
    
    Args:
        config: Transport configuration
        tool_name: Name of the tool to call
        arguments: Tool arguments
        
    Yields:
        Streaming tool execution results
    """
    client = MCPUnifiedClient(config)
    async for result in client.call_tool_streaming(tool_name, arguments):
        yield result


# Helper functions for creating configurations
def create_stdio_config(command: List[str], **kwargs) -> UnifiedTransportConfig:
    """
    Create configuration for stdio transport.
    
    Args:
        command: Command and arguments to run
        **kwargs: Additional configuration options
        
    Returns:
        Transport configuration
    """
    return UnifiedTransportConfig(
        transport_type="stdio",
        command=command,
        enable_streaming=True,
        **kwargs
    )


def create_http_config(url: str, headers: Optional[Dict[str, str]] = None, **kwargs) -> UnifiedTransportConfig:
    """
    Create configuration for HTTP transport.
    
    Args:
        url: Server URL
        headers: Optional HTTP headers
        **kwargs: Additional configuration options
        
    Returns:
        Transport configuration
    """
    return UnifiedTransportConfig(
        transport_type="http",
        url=url,
        headers=headers,
        enable_streaming=True,
        **kwargs
    )


def create_streamable_http_config(url: str, headers: Optional[Dict[str, str]] = None, **kwargs) -> UnifiedTransportConfig:
    """
    Create configuration for streamable HTTP transport.
    
    Args:
        url: Server URL
        headers: Optional HTTP headers
        **kwargs: Additional configuration options
        
    Returns:
        Transport configuration
    """
    return UnifiedTransportConfig(
        transport_type="streamable_http",
        url=url,
        headers=headers,
        enable_streaming=True,
        **kwargs
    )


def create_sse_config(url: str, headers: Optional[Dict[str, str]] = None, **kwargs) -> UnifiedTransportConfig:
    """
    Create configuration for SSE transport.
    
    Args:
        url: Server URL
        headers: Optional HTTP headers
        **kwargs: Additional configuration options
        
    Returns:
        Transport configuration
    """
    return UnifiedTransportConfig(
        transport_type="sse",
        url=url,
        headers=headers,
        enable_streaming=True,
        **kwargs
    )


def create_auto_config(url: str, **kwargs) -> UnifiedTransportConfig:
    """
    Create configuration with auto-detection.
    
    Args:
        url: Server URL or command
        **kwargs: Additional configuration options
        
    Returns:
        Transport configuration
    """
    return UnifiedTransportConfig(
        transport_type="auto",
        url=url,
        auto_detect=True,
        enable_streaming=True,
        **kwargs
    )


# Example usage
async def example_unified_usage():
    """Example of how to use the unified MCP client with streaming support."""
    
    # Example 1: Auto-detection from URL with streaming
    config1 = create_auto_config("http://localhost:8000/mcp")
    client1 = MCPUnifiedClient(config1)
    
    # Example 2: Explicit stdio transport with streaming
    config2 = create_stdio_config(["python", "path/to/mcp/server.py"])
    client2 = MCPUnifiedClient(config2)
    
    # Example 3: Explicit streamable HTTP transport with streaming
    config3 = create_streamable_http_config("http://localhost:8001/mcp")
    client3 = MCPUnifiedClient(config3)
    
    # Get tools from different transports
    try:
        tools1 = await client1.get_tools()
        print(f"Auto-detected transport tools: {len(tools1)}")
        
        tools2 = await client2.get_tools()
        print(f"STDIO transport tools: {len(tools2)}")
        
        tools3 = await client3.get_tools()
        print(f"Streamable HTTP transport tools: {len(tools3)}")
        
        # Example streaming tool call
        if tools1:
            tool_name = tools1[0]["function"]["name"]
            print(f"Calling tool with streaming: {tool_name}")
            
            async for result in client1.call_tool_streaming(tool_name, {}):
                print(f"Streaming result: {result}")
        
    except Exception as e:
        logger.error(f"Error getting tools: {e}")


# Export constants for availability checking
MCP_STREAMING_AVAILABLE = MCP_AVAILABLE and STREAMABLE_HTTP_AVAILABLE

# Export all public functions and classes
__all__ = [
    "MCPUnifiedClient",
    "UnifiedTransportConfig", 
    "create_auto_config",
    "create_http_config",
    "create_streamable_http_config",
    "create_stdio_config",
    "create_sse_config",
    "MCP_STREAMING_AVAILABLE",
    "STREAMABLE_HTTP_AVAILABLE",
    "HTTPX_AVAILABLE",
    "MCP_AVAILABLE",
    "call_tool_streaming_sync",
    "execute_tool_call_streaming_unified",
]


if __name__ == "__main__":
    # Run example
    asyncio.run(example_unified_usage()) 
