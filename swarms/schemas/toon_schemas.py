"""
TOON (Token-Oriented Object Notation) Schema Definitions

This module defines Pydantic schemas for TOON SDK integration, enabling
compact, human-readable JSON serialization optimized for LLM prompts.

TOON provides 30-60% token reduction compared to standard JSON while
maintaining readability and schema-awareness.

References:
    - TOON Spec: https://github.com/toon-format
    - Benchmarks: 73.9% retrieval accuracy for tables, 69.7% for varying fields
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class TOONConnection(BaseModel):
    """
    Configuration for connecting to TOON SDK services.

    This schema follows the same pattern as MCPConnection but is
    optimized for TOON-specific serialization and deserialization.

    Attributes:
        type: Connection type identifier (always 'toon')
        url: TOON SDK endpoint URL
        api_key: Authentication API key
        serialization_format: Output format ('toon', 'json', 'compact')
        enable_compression: Enable automatic token compression
        schema_aware: Use schema information for better compression
        transport: Transport protocol ('http', 'https')
        headers: Additional HTTP headers
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts for failed requests
        retry_backoff: Backoff multiplier for retries

    Examples:
        >>> connection = TOONConnection(
        ...     url="https://api.toon-format.com/v1",
        ...     api_key="toon_key_xxx",
        ...     serialization_format="toon",
        ...     enable_compression=True
        ... )
    """

    type: Optional[str] = Field(
        default="toon",
        description="Connection type identifier, always 'toon'",
    )
    url: Optional[str] = Field(
        default="https://api.toon-format.com/v1",
        description="TOON SDK API endpoint URL",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Authentication API key for TOON SDK",
    )
    serialization_format: Optional[
        Literal["toon", "json", "compact"]
    ] = Field(
        default="toon",
        description="Output serialization format: 'toon' (compact), 'json' (standard), or 'compact' (minimal)",
    )
    enable_compression: Optional[bool] = Field(
        default=True,
        description="Enable automatic token compression (30-60% reduction)",
    )
    schema_aware: Optional[bool] = Field(
        default=True,
        description="Use schema information for optimized serialization",
    )
    transport: Optional[str] = Field(
        default="https",
        description="Transport protocol: 'http' or 'https'",
    )
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Additional HTTP headers for requests",
    )
    timeout: Optional[int] = Field(
        default=30,
        description="Request timeout in seconds",
    )
    max_retries: Optional[int] = Field(
        default=3,
        description="Maximum retry attempts for failed requests",
    )
    retry_backoff: Optional[float] = Field(
        default=2.0,
        description="Exponential backoff multiplier for retries",
    )
    tool_configurations: Optional[Dict[Any, Any]] = Field(
        default=None,
        description="Configuration settings for TOON tools",
    )

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


class TOONSerializationOptions(BaseModel):
    """
    Fine-grained options for TOON serialization behavior.

    These options control how JSON data is converted to TOON format,
    allowing customization for specific use cases.

    Attributes:
        compact_keys: Use abbreviated key names
        omit_null_values: Exclude null/None values from output
        flatten_nested: Flatten nested structures where possible
        preserve_order: Maintain original key ordering
        indent_level: Indentation spaces (0 for single-line)
        use_shorthand: Enable TOON shorthand syntax
        max_depth: Maximum nesting depth before flattening
        array_compression: Compress repetitive array structures

    Examples:
        >>> options = TOONSerializationOptions(
        ...     compact_keys=True,
        ...     omit_null_values=True,
        ...     indent_level=0
        ... )
    """

    compact_keys: Optional[bool] = Field(
        default=True,
        description="Use abbreviated key names for common fields",
    )
    omit_null_values: Optional[bool] = Field(
        default=True,
        description="Exclude null/None values from serialized output",
    )
    flatten_nested: Optional[bool] = Field(
        default=False,
        description="Flatten nested structures where semantically safe",
    )
    preserve_order: Optional[bool] = Field(
        default=True,
        description="Maintain original key ordering in output",
    )
    indent_level: Optional[int] = Field(
        default=0,
        ge=0,
        le=8,
        description="Indentation spaces (0 for compact single-line)",
    )
    use_shorthand: Optional[bool] = Field(
        default=True,
        description="Enable TOON shorthand syntax for common patterns",
    )
    max_depth: Optional[int] = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum nesting depth before flattening",
    )
    array_compression: Optional[bool] = Field(
        default=True,
        description="Compress repetitive array structures",
    )

    class Config:
        extra = "allow"


class TOONToolDefinition(BaseModel):
    """
    Definition of a TOON-compatible tool/function.

    This schema describes a tool that can serialize its inputs/outputs
    using TOON format for optimal token efficiency.

    Attributes:
        name: Unique tool identifier
        description: Human-readable tool description
        input_schema: JSON Schema for input parameters
        output_schema: JSON Schema for output data
        requires_toon_serialization: Whether tool uses TOON format
        serialization_options: Custom TOON serialization settings
        compression_ratio: Expected token reduction percentage
        category: Tool category for organization
        version: Tool version string

    Examples:
        >>> tool = TOONToolDefinition(
        ...     name="get_user_data",
        ...     description="Fetch user profile data",
        ...     input_schema={"type": "object", "properties": {...}},
        ...     requires_toon_serialization=True
        ... )
    """

    name: str = Field(
        description="Unique identifier for the tool"
    )
    description: Optional[str] = Field(
        default="",
        description="Human-readable description of tool functionality",
    )
    input_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON Schema defining input parameters",
    )
    output_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON Schema defining output data structure",
    )
    requires_toon_serialization: Optional[bool] = Field(
        default=True,
        description="Whether this tool requires TOON format serialization",
    )
    serialization_options: Optional[TOONSerializationOptions] = Field(
        default=None,
        description="Custom TOON serialization options for this tool",
    )
    compression_ratio: Optional[float] = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Expected token reduction ratio (0.0-1.0, e.g., 0.45 = 45% reduction)",
    )
    category: Optional[str] = Field(
        default="general",
        description="Tool category (e.g., 'data', 'compute', 'io')",
    )
    version: Optional[str] = Field(
        default="1.0.0",
        description="Tool version string (semantic versioning)",
    )

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


class TOONRequest(BaseModel):
    """
    Request payload for TOON SDK API calls.

    This schema structures data for encoding, decoding, or tool
    execution requests to the TOON SDK.

    Attributes:
        operation: Operation type ('encode', 'decode', 'validate')
        data: Input data to process
        schema: Optional JSON Schema for validation
        options: Serialization options
        format: Desired output format
        metadata: Additional request metadata

    Examples:
        >>> request = TOONRequest(
        ...     operation="encode",
        ...     data={"user": "Alice", "age": 30},
        ...     format="toon"
        ... )
    """

    operation: Literal["encode", "decode", "validate", "convert"] = Field(
        description="Operation to perform: 'encode' (JSON→TOON), 'decode' (TOON→JSON), 'validate', or 'convert'"
    )
    data: Union[Dict[str, Any], str, List[Any]] = Field(
        description="Input data to process (JSON object, TOON string, or array)"
    )
    schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional JSON Schema for validation and optimization",
    )
    options: Optional[TOONSerializationOptions] = Field(
        default=None,
        description="Serialization options for this request",
    )
    format: Optional[Literal["toon", "json", "compact"]] = Field(
        default="toon",
        description="Desired output format",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional request metadata",
    )

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


class TOONResponse(BaseModel):
    """
    Response from TOON SDK API calls.

    This schema structures the response from encoding, decoding,
    or validation operations.

    Attributes:
        operation: Original operation type
        status: Response status ('success', 'error', 'partial')
        result: Processed data (encoded TOON or decoded JSON)
        original_tokens: Token count before processing
        compressed_tokens: Token count after TOON encoding
        compression_ratio: Actual compression ratio achieved
        metadata: Additional response metadata
        errors: List of errors if status is 'error' or 'partial'
        warnings: Non-critical warnings
        execution_time_ms: Processing time in milliseconds

    Examples:
        >>> response = TOONResponse(
        ...     operation="encode",
        ...     status="success",
        ...     result="usr:Alice age:30",
        ...     original_tokens=15,
        ...     compressed_tokens=8,
        ...     compression_ratio=0.47
        ... )
    """

    operation: str = Field(
        description="Operation that was performed"
    )
    status: Literal["success", "error", "partial"] = Field(
        description="Response status indicator"
    )
    result: Union[str, Dict[str, Any], List[Any]] = Field(
        description="Processed data (TOON string, JSON object, or array)"
    )
    original_tokens: Optional[int] = Field(
        default=None,
        description="Token count of original input",
    )
    compressed_tokens: Optional[int] = Field(
        default=None,
        description="Token count after TOON compression",
    )
    compression_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Compression ratio achieved (0.0-1.0)",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional response metadata",
    )
    errors: Optional[List[str]] = Field(
        default=None,
        description="List of error messages if status is 'error' or 'partial'",
    )
    warnings: Optional[List[str]] = Field(
        default=None,
        description="Non-critical warnings during processing",
    )
    execution_time_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Processing time in milliseconds",
    )

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


class MultipleTOONConnections(BaseModel):
    """
    Container for multiple TOON SDK connections.

    Allows managing multiple TOON endpoints with different
    configurations simultaneously.

    Attributes:
        connections: List of TOONConnection objects

    Examples:
        >>> connections = MultipleTOONConnections(
        ...     connections=[
        ...         TOONConnection(url="https://api1.toon.com", api_key="key1"),
        ...         TOONConnection(url="https://api2.toon.com", api_key="key2")
        ...     ]
        ... )
    """

    connections: List[TOONConnection] = Field(
        description="List of TOON SDK connections"
    )

    class Config:
        arbitrary_types_allowed = True
