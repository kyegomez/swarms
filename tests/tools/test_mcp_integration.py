
import pytest
import asyncio
from swarms.tools.mcp_integration import (
    MCPServer,
    MCPServerStdio,
    MCPServerSse,
    mcp_flow,
    mcp_flow_get_tool_schema,
    batch_mcp_flow
)

# Test basic server connectivity
def test_server_connection():
    """
    Test that a user can connect to the MCP server successfully
    """
    params = {"url": "http://localhost:8000"}
    server = MCPServerSse(params, cache_tools_list=True)
    
    # Connect should work
    asyncio.run(server.connect())
    assert server.session is not None
    
    # Cleanup should work
    asyncio.run(server.cleanup())
    assert server.session is None

# Test tool listing functionality
def test_list_tools():
    """
    Test that a user can retrieve available tools from the server
    """
    params = {"url": "http://localhost:8000"}
    server = MCPServerSse(params)
    
    asyncio.run(server.connect())
    tools = asyncio.run(server.list_tools())
    
    assert isinstance(tools, list)
    assert len(tools) > 0
    
    asyncio.run(server.cleanup())

# Test tool execution
def test_tool_execution():
    """
    Test that a user can execute a tool successfully
    """
    params = {"url": "http://localhost:8000"}
    function_call = {
        "tool_name": "add",
        "arguments": {"a": 5, "b": 3}
    }
    
    result = mcp_flow(params, function_call)
    assert result is not None

# Test batch operations
def test_batch_execution():
    """
    Test that a user can execute multiple tools in batch
    """
    params_list = [
        {"url": "http://localhost:8000"},
        {"url": "http://localhost:8000"}
    ]
    function_calls = [
        {"tool_name": "add", "arguments": {"a": 1, "b": 2}},
        {"tool_name": "subtract", "arguments": {"a": 5, "b": 3}}
    ]
    
    results = batch_mcp_flow(params_list, function_calls)
    assert len(results) == 2
    assert all(result is not None for result in results)

# Test error handling
def test_error_handling():
    """
    Test that users receive proper error messages for invalid operations
    """
    params = {"url": "http://localhost:8000"}
    invalid_function = {
        "tool_name": "nonexistent_tool",
        "arguments": {}
    }
    
    with pytest.raises(Exception):
        mcp_flow(params, invalid_function)

# Test tool schema retrieval
def test_get_tool_schema():
    """
    Test that users can retrieve tool schemas correctly
    """
    params = {"url": "http://localhost:8000"}
    schema = mcp_flow_get_tool_schema(params)
    
    assert isinstance(schema, dict)
    assert "tools" in schema or "functions" in schema

# Test server reconnection
def test_server_reconnection():
    """
    Test that users can reconnect to the server after disconnection
    """
    params = {"url": "http://localhost:8000"}
    server = MCPServerSse(params)
    
    # First connection
    asyncio.run(server.connect())
    asyncio.run(server.cleanup())
    
    # Second connection should work
    asyncio.run(server.connect())
    assert server.session is not None
    asyncio.run(server.cleanup())

# Test cache functionality
def test_cache_behavior():
    """
    Test that tool caching works as expected for users
    """
    params = {"url": "http://localhost:8000"}
    server = MCPServerSse(params, cache_tools_list=True)
    
    asyncio.run(server.connect())
    
    # First call should cache
    tools1 = asyncio.run(server.list_tools())
    # Second call should use cache
    tools2 = asyncio.run(server.list_tools())
    
    assert tools1 == tools2
    
    asyncio.run(server.cleanup())
