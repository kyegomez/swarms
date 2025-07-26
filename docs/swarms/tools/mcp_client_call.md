# MCP Client Call Reference Documentation

This document provides a comprehensive reference for the MCP (Model Control Protocol) client call functions, including detailed parameter descriptions, return types, and usage examples.

## Table of Contents

- [aget_mcp_tools](#aget_mcp_tools)

- [get_mcp_tools_sync](#get_mcp_tools_sync)

- [get_tools_for_multiple_mcp_servers](#get_tools_for_multiple_mcp_servers)

- [execute_tool_call_simple](#execute_tool_call_simple)

## Function Reference

### aget_mcp_tools

Asynchronously fetches available MCP tools from the server with retry logic.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| server_path | Optional[str] | No | Path to the MCP server script |
| format | str | No | Format of the returned tools (default: "openai") |
| connection | Optional[MCPConnection] | No | MCP connection object |
| *args | Any | No | Additional positional arguments |
| **kwargs | Any | No | Additional keyword arguments |

#### Returns

- `List[Dict[str, Any]]`: List of available MCP tools in OpenAI format

#### Raises

- `MCPValidationError`: If server_path is invalid

- `MCPConnectionError`: If connection to server fails

#### Example

```python
import asyncio
from swarms.tools.mcp_client_call import aget_mcp_tools
from swarms.tools.mcp_connection import MCPConnection

async def main():
    # Using server path
    tools = await aget_mcp_tools(server_path="http://localhost:8000")
    
    # Using connection object
    connection = MCPConnection(
        host="localhost",
        port=8000,
        headers={"Authorization": "Bearer token"}
    )
    tools = await aget_mcp_tools(connection=connection)
    
    print(f"Found {len(tools)} tools")

if __name__ == "__main__":
    asyncio.run(main())
```

### get_mcp_tools_sync

Synchronous version of get_mcp_tools that handles event loop management.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| server_path | Optional[str] | No | Path to the MCP server script |
| format | str | No | Format of the returned tools (default: "openai") |
| connection | Optional[MCPConnection] | No | MCP connection object |
| *args | Any | No | Additional positional arguments |
| **kwargs | Any | No | Additional keyword arguments |

#### Returns

- `List[Dict[str, Any]]`: List of available MCP tools in OpenAI format

#### Raises

- `MCPValidationError`: If server_path is invalid

- `MCPConnectionError`: If connection to server fails

- `MCPExecutionError`: If event loop management fails

#### Example

```python
from swarms.tools.mcp_client_call import get_mcp_tools_sync
from swarms.tools.mcp_connection import MCPConnection

# Using server path
tools = get_mcp_tools_sync(server_path="http://localhost:8000")

# Using connection object
connection = MCPConnection(
    host="localhost",
    port=8000,
    headers={"Authorization": "Bearer token"}
)
tools = get_mcp_tools_sync(connection=connection)

print(f"Found {len(tools)} tools")
```

### get_tools_for_multiple_mcp_servers

Get tools for multiple MCP servers concurrently using ThreadPoolExecutor.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| urls | List[str] | Yes | List of server URLs to fetch tools from |
| connections | List[MCPConnection] | No | Optional list of MCPConnection objects |
| format | str | No | Format to return tools in (default: "openai") |
| output_type | Literal["json", "dict", "str"] | No | Type of output format (default: "str") |
| max_workers | Optional[int] | No | Maximum number of worker threads |

#### Returns

- `List[Dict[str, Any]]`: Combined list of tools from all servers

#### Raises

- `MCPExecutionError`: If fetching tools from any server fails

#### Example

```python
from swarms.tools.mcp_client_call import get_tools_for_multiple_mcp_servers
from swarms.tools.mcp_connection import MCPConnection

# Define server URLs
urls = [
    "http://server1:8000",
    "http://server2:8000"
]

# Optional: Define connections
connections = [
    MCPConnection(host="server1", port=8000),
    MCPConnection(host="server2", port=8000)
]

# Get tools from all servers
tools = get_tools_for_multiple_mcp_servers(
    urls=urls,
    connections=connections,
    format="openai",
    output_type="dict",
    max_workers=4
)

print(f"Found {len(tools)} tools across all servers")
```

### execute_tool_call_simple

Execute a tool call using the MCP client.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| response | Any | No | Tool call response object |
| server_path | str | No | Path to the MCP server |
| connection | Optional[MCPConnection] | No | MCP connection object |
| output_type | Literal["json", "dict", "str", "formatted"] | No | Type of output format (default: "str") |
| *args | Any | No | Additional positional arguments |
| **kwargs | Any | No | Additional keyword arguments |

#### Returns

- `List[Dict[str, Any]]`: Result of the tool execution

#### Raises

- `MCPConnectionError`: If connection to server fails

- `MCPExecutionError`: If tool execution fails

#### Example
```python
import asyncio
from swarms.tools.mcp_client_call import execute_tool_call_simple
from swarms.tools.mcp_connection import MCPConnection

async def main():
    # Example tool call response
    response = {
        "name": "example_tool",
        "parameters": {"param1": "value1"}
    }
    
    # Using server path
    result = await execute_tool_call_simple(
        response=response,
        server_path="http://localhost:8000",
        output_type="json"
    )
    
    # Using connection object
    connection = MCPConnection(
        host="localhost",
        port=8000,
        headers={"Authorization": "Bearer token"}
    )
    result = await execute_tool_call_simple(
        response=response,
        connection=connection,
        output_type="dict"
    )
    
    print(f"Tool execution result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Error Handling

The MCP client functions use a retry mechanism with exponential backoff for failed requests. The following error types may be raised:

- `MCPValidationError`: Raised when input validation fails

- `MCPConnectionError`: Raised when connection to the MCP server fails

- `MCPExecutionError`: Raised when tool execution fails

## Best Practices

1. Always handle potential exceptions when using these functions
2. Use connection objects for authenticated requests
3. Consider using the async versions for better performance in async applications
4. Use appropriate output types based on your needs
5. When working with multiple servers, adjust max_workers based on your system's capabilities

