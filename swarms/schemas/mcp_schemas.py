from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Literal, Callable


class MCPConnection(BaseModel):
    """
    Configuration for MCP (Model Context Protocol) connections.
    
    This schema supports multiple transport types including stdio, http, 
    streamable_http, and sse. All transport types are optional and can be
    configured based on requirements. Includes streaming support for real-time communication.
    """
    
    type: Optional[str] = Field(
        default="mcp",
        description="The type of connection, defaults to 'mcp'",
    )
    
    url: Optional[str] = Field(
        default="http://localhost:8000/mcp",
        description="The URL endpoint for the MCP server or command path for stdio",
    )
    
    transport: Optional[Literal["stdio", "http", "streamable_http", "sse", "auto"]] = Field(
        default="streamable_http",
        description="The transport protocol to use for the MCP server. 'auto' enables auto-detection.",
    )
    
    # STDIO specific
    command: Optional[List[str]] = Field(
        default=None,
        description="Command and arguments for stdio transport",
    )
    
    # HTTP specific
    headers: Optional[Dict[str, str]] = Field(
        default=None, 
        description="Headers to send to the MCP server"
    )
    
    authorization_token: Optional[str] = Field(
        default=None,
        description="Authentication token for accessing the MCP server",
    )
    
    timeout: Optional[int] = Field(
        default=10, 
        description="Timeout for the MCP server in seconds"
    )
    
    # Auto-detection settings
    auto_detect: Optional[bool] = Field(
        default=True,
        description="Whether to auto-detect transport type from URL"
    )
    
    fallback_transport: Optional[Literal["stdio", "http", "streamable_http", "sse"]] = Field(
        default="sse",
        description="Fallback transport if auto-detection fails"
    )
    
    # Streaming settings
    enable_streaming: Optional[bool] = Field(
        default=True,
        description="Whether to enable streaming support for real-time communication"
    )
    
    streaming_timeout: Optional[int] = Field(
        default=None,
        description="Timeout for streaming operations in seconds"
    )
    
    streaming_callback: Optional[Callable[[str], None]] = Field(
        default=None,
        description="Callback function for streaming chunks"
    )
    
    # Tool configurations
    tool_configurations: Optional[Dict[Any, Any]] = Field(
        default=None,
        description="Dictionary containing configuration settings for MCP tools",
    )

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


class MultipleMCPConnections(BaseModel):
    """
    Configuration for multiple MCP connections.
    
    This allows managing multiple MCP servers with different transport types
    and configurations simultaneously. Includes streaming support.
    """
    
    connections: List[MCPConnection] = Field(
        default=[], 
        description="List of MCP connections"
    )
    
    # Global settings for multiple connections
    max_concurrent: Optional[int] = Field(
        default=None,
        description="Maximum number of concurrent connections"
    )
    
    retry_attempts: Optional[int] = Field(
        default=3,
        description="Number of retry attempts for failed connections"
    )
    
    retry_delay: Optional[float] = Field(
        default=1.0,
        description="Delay between retry attempts in seconds"
    )
    
    # Global streaming settings
    enable_streaming: Optional[bool] = Field(
        default=True,
        description="Whether to enable streaming support globally"
    )

    class Config:
        arbitrary_types_allowed = True


class MCPToolConfig(BaseModel):
    """
    Configuration for individual MCP tools.
    
    This allows fine-grained control over tool behavior and settings.
    Includes streaming support for individual tools.
    """
    
    name: str = Field(
        description="Name of the tool"
    )
    
    description: Optional[str] = Field(
        default=None,
        description="Description of the tool"
    )
    
    enabled: bool = Field(
        default=True,
        description="Whether the tool is enabled"
    )
    
    timeout: Optional[int] = Field(
        default=None,
        description="Tool-specific timeout in seconds"
    )
    
    retry_attempts: Optional[int] = Field(
        default=None,
        description="Tool-specific retry attempts"
    )
    
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Tool-specific parameters"
    )
    
    # Tool-specific streaming settings
    enable_streaming: Optional[bool] = Field(
        default=True,
        description="Whether to enable streaming for this specific tool"
    )
    
    streaming_timeout: Optional[int] = Field(
        default=None,
        description="Tool-specific streaming timeout in seconds"
    )

    class Config:
        arbitrary_types_allowed = True


class MCPTransportConfig(BaseModel):
    """
    Detailed transport configuration for MCP connections.
    
    This provides advanced configuration options for each transport type.
    Includes comprehensive streaming support.
    """
    
    transport_type: Literal["stdio", "http", "streamable_http", "sse", "auto"] = Field(
        description="The transport type to use"
    )
    
    # Connection settings
    url: Optional[str] = Field(
        default=None,
        description="URL for HTTP-based transports or command path for stdio"
    )
    
    command: Optional[List[str]] = Field(
        default=None,
        description="Command and arguments for stdio transport"
    )
    
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="HTTP headers for HTTP-based transports"
    )
    
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
    
    fallback_transport: Literal["stdio", "http", "streamable_http", "sse"] = Field(
        default="sse",
        description="Fallback transport if auto-detection fails"
    )
    
    # Advanced settings
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts"
    )
    
    retry_delay: float = Field(
        default=1.0,
        description="Delay between retry attempts in seconds"
    )
    
    keep_alive: bool = Field(
        default=True,
        description="Whether to keep the connection alive"
    )
    
    verify_ssl: bool = Field(
        default=True,
        description="Whether to verify SSL certificates for HTTPS connections"
    )
    
    # Streaming settings
    enable_streaming: bool = Field(
        default=True,
        description="Whether to enable streaming support"
    )
    
    streaming_timeout: Optional[int] = Field(
        default=None,
        description="Timeout for streaming operations in seconds"
    )
    
    streaming_buffer_size: Optional[int] = Field(
        default=1024,
        description="Buffer size for streaming operations"
    )
    
    streaming_chunk_size: Optional[int] = Field(
        default=1024,
        description="Chunk size for streaming operations"
    )

    class Config:
        arbitrary_types_allowed = True


class MCPErrorResponse(BaseModel):
    """
    Standardized error response for MCP operations.
    """
    
    error: str = Field(
        description="Error message"
    )
    
    error_type: str = Field(
        description="Type of error (e.g., 'connection', 'timeout', 'validation')"
    )
    
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    
    timestamp: Optional[str] = Field(
        default=None,
        description="Timestamp when the error occurred"
    )

    class Config:
        arbitrary_types_allowed = True


class MCPToolCall(BaseModel):
    """
    Standardized tool call request.
    """
    
    tool_name: str = Field(
        description="Name of the tool to call"
    )
    
    arguments: Dict[str, Any] = Field(
        default={},
        description="Arguments to pass to the tool"
    )
    
    timeout: Optional[int] = Field(
        default=None,
        description="Timeout for this specific tool call"
    )
    
    retry_attempts: Optional[int] = Field(
        default=None,
        description="Retry attempts for this specific tool call"
    )
    
    # Streaming settings for tool calls
    enable_streaming: Optional[bool] = Field(
        default=True,
        description="Whether to enable streaming for this tool call"
    )
    
    streaming_timeout: Optional[int] = Field(
        default=None,
        description="Timeout for streaming operations in this tool call"
    )

    class Config:
        arbitrary_types_allowed = True


class MCPToolResult(BaseModel):
    """
    Standardized tool call result.
    """
    
    success: bool = Field(
        description="Whether the tool call was successful"
    )
    
    result: Optional[Any] = Field(
        default=None,
        description="Result of the tool call"
    )
    
    error: Optional[str] = Field(
        default=None,
        description="Error message if the call failed"
    )
    
    execution_time: Optional[float] = Field(
        default=None,
        description="Execution time in seconds"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata about the execution"
    )
    
    # Streaming result metadata
    is_streaming: Optional[bool] = Field(
        default=False,
        description="Whether this result is from a streaming operation"
    )
    
    stream_chunk: Optional[int] = Field(
        default=None,
        description="Chunk number for streaming results"
    )
    
    stream_complete: Optional[bool] = Field(
        default=False,
        description="Whether the streaming operation is complete"
    )

    class Config:
        arbitrary_types_allowed = True


class MCPStreamingConfig(BaseModel):
    """
    Configuration for MCP streaming operations.
    """
    
    enable_streaming: bool = Field(
        default=True,
        description="Whether to enable streaming support"
    )
    
    streaming_timeout: Optional[int] = Field(
        default=None,
        description="Timeout for streaming operations in seconds"
    )
    
    buffer_size: int = Field(
        default=1024,
        description="Buffer size for streaming operations"
    )
    
    chunk_size: int = Field(
        default=1024,
        description="Chunk size for streaming operations"
    )
    
    max_stream_duration: Optional[int] = Field(
        default=None,
        description="Maximum duration for streaming operations in seconds"
    )
    
    enable_compression: bool = Field(
        default=False,
        description="Whether to enable compression for streaming"
    )
    
    compression_level: int = Field(
        default=6,
        description="Compression level (1-9)"
    )

    class Config:
        arbitrary_types_allowed = True


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

    class Config:
        arbitrary_types_allowed = True
