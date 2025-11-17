from swarms.tools.mcp_client_tools import (
    execute_tool_call_simple,
    get_mcp_tools_sync,
)


async def main():
    # Prepare the tool call in OpenAI-compatible format
    response = {
        "function": {"name": "greet", "arguments": {"name": "Alice"}}
    }
    result = await execute_tool_call_simple(
        server_path="http://localhost:8000/mcp",
        response=response,
        # transport="streamable_http",
    )
    print("Tool call result:", result)
    return result


if __name__ == "__main__":
    print(get_mcp_tools_sync(server_path="http://localhost:8000/mcp"))

    import asyncio

    asyncio.run(main())
