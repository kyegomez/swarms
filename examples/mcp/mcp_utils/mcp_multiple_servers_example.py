"""
Example demonstrating how to execute multiple tools across multiple MCP servers.

This example shows how to:
1. Create a mapping of function names to servers
2. Execute multiple tool calls across different servers
3. Handle responses with tool calls and route them to the appropriate servers
"""

import asyncio
from swarms.tools.mcp_client_tools import (
    execute_multiple_tools_on_multiple_mcp_servers,
    execute_multiple_tools_on_multiple_mcp_servers_sync,
    get_tools_for_multiple_mcp_servers,
)
from swarms.schemas.mcp_schemas import MCPConnection


def example_sync_execution():
    """Example of synchronous execution across multiple MCP servers."""

    # Example server URLs (replace with your actual MCP server URLs)
    urls = [
        "http://localhost:8000/sse",  # Server 1
        "http://localhost:8001/sse",  # Server 2
        "http://localhost:8002/sse",  # Server 3
    ]

    # Optional: Create connection objects for each server
    connections = [
        MCPConnection(
            url="http://localhost:8000/sse",
            authorization_token="token1",  # if needed
            timeout=10,
        ),
        MCPConnection(
            url="http://localhost:8001/sse",
            authorization_token="token2",  # if needed
            timeout=10,
        ),
        MCPConnection(
            url="http://localhost:8002/sse",
            authorization_token="token3",  # if needed
            timeout=10,
        ),
    ]

    # Example responses containing tool calls
    # These would typically come from an LLM that decided to use tools
    responses = [
        {
            "function": {
                "name": "search_web",
                "arguments": {
                    "query": "python programming best practices"
                },
            }
        },
        {
            "function": {
                "name": "search_database",
                "arguments": {"table": "users", "id": 123},
            }
        },
        {
            "function": {
                "name": "send_email",
                "arguments": {
                    "to": "user@example.com",
                    "subject": "Test email",
                    "body": "This is a test email",
                },
            }
        },
    ]

    print("=== Synchronous Execution Example ===")
    print(
        f"Executing {len(responses)} tool calls across {len(urls)} servers..."
    )

    try:
        # Execute all tool calls across multiple servers
        results = execute_multiple_tools_on_multiple_mcp_servers_sync(
            responses=responses,
            urls=urls,
            connections=connections,
            output_type="dict",
            max_concurrent=5,  # Limit concurrent executions
        )

        print(f"\nExecution completed! Got {len(results)} results:")
        for i, result in enumerate(results):
            print(f"\nResult {i + 1}:")
            print(f"  Function: {result['function_name']}")
            print(f"  Server: {result['server_url']}")
            print(f"  Status: {result['status']}")
            if result["status"] == "success":
                print(f"  Result: {result['result']}")
            else:
                print(
                    f"  Error: {result.get('error', 'Unknown error')}"
                )

    except Exception as e:
        print(f"Error during execution: {str(e)}")


async def example_async_execution():
    """Example of asynchronous execution across multiple MCP servers."""

    # Example server URLs
    urls = [
        "http://localhost:8000/sse",
        "http://localhost:8001/sse",
        "http://localhost:8002/sse",
    ]

    # Example responses with multiple tool calls in a single response
    responses = [
        {
            "tool_calls": [
                {
                    "function": {
                        "name": "search_web",
                        "arguments": {
                            "query": "machine learning trends 2024"
                        },
                    }
                },
                {
                    "function": {
                        "name": "search_database",
                        "arguments": {
                            "table": "articles",
                            "category": "AI",
                        },
                    }
                },
            ]
        },
        {
            "function": {
                "name": "send_notification",
                "arguments": {
                    "user_id": 456,
                    "message": "Your analysis is complete",
                },
            }
        },
    ]

    print("\n=== Asynchronous Execution Example ===")
    print(
        f"Executing tool calls across {len(urls)} servers asynchronously..."
    )

    try:
        # Execute all tool calls across multiple servers
        results = (
            await execute_multiple_tools_on_multiple_mcp_servers(
                responses=responses,
                urls=urls,
                output_type="str",
                max_concurrent=3,
            )
        )

        print(
            f"\nAsync execution completed! Got {len(results)} results:"
        )
        for i, result in enumerate(results):
            print(f"\nResult {i + 1}:")
            print(f"  Response Index: {result['response_index']}")
            print(f"  Function: {result['function_name']}")
            print(f"  Server: {result['server_url']}")
            print(f"  Status: {result['status']}")
            if result["status"] == "success":
                print(f"  Result: {result['result']}")
            else:
                print(
                    f"  Error: {result.get('error', 'Unknown error')}"
                )

    except Exception as e:
        print(f"Error during async execution: {str(e)}")


def example_get_tools_from_multiple_servers():
    """Example of getting tools from multiple servers."""

    urls = [
        "http://localhost:8000/sse",
        "http://localhost:8001/sse",
        "http://localhost:8002/sse",
    ]

    print("\n=== Getting Tools from Multiple Servers ===")

    try:
        # Get all available tools from all servers
        all_tools = get_tools_for_multiple_mcp_servers(
            urls=urls, format="openai", output_type="dict"
        )

        print(
            f"Found {len(all_tools)} total tools across all servers:"
        )

        # Group tools by function name to see what's available
        function_names = set()
        for tool in all_tools:
            if isinstance(tool, dict) and "function" in tool:
                function_names.add(tool["function"]["name"])
            elif hasattr(tool, "name"):
                function_names.add(tool.name)

        print("Available functions:")
        for func_name in sorted(function_names):
            print(f"  - {func_name}")

    except Exception as e:
        print(f"Error getting tools: {str(e)}")


if __name__ == "__main__":
    # Run synchronous example
    example_sync_execution()

    # Run async example
    asyncio.run(example_async_execution())

    # Get tools from multiple servers
    example_get_tools_from_multiple_servers()
