import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from swarms.structs.aop import AOPCluster
from swarms.tools.mcp_client_tools import execute_tool_call_simple


async def discover_agents_example():
    """
    Discover all agents using the AOPCluster and print the result.
    """
    aop_cluster = AOPCluster(
        urls=["http://localhost:5932/mcp"],
        transport="streamable-http",
    )
    tool = aop_cluster.find_tool_by_server_name("discover_agents")
    if not tool:
        print("discover_agents tool not found.")
        return None

    tool_call_request = {
        "type": "function",
        "function": {
            "name": "discover_agents",
            "arguments": "{}",
        },
    }

    result = await execute_tool_call_simple(
        response=tool_call_request,
        server_path="http://localhost:5932/mcp",
        output_type="dict",
        verbose=False,
    )
    print(json.dumps(result, indent=2))
    return result


async def raw_mcp_discover_agents_example():
    """
    Call the MCP server directly using the raw MCP client to execute the
    built-in "discover_agents" tool and print the JSON result.

    This demonstrates how to:
    - Initialize an MCP client over streamable HTTP
    - List available tools (optional)
    - Call a specific tool by name with arguments
    """
    url = "http://localhost:5932/mcp"

    # Open a raw MCP client connection
    async with streamablehttp_client(url, timeout=10) as ctx:
        if len(ctx) == 2:
            read, write = ctx
        else:
            read, write, *_ = ctx

        async with ClientSession(read, write) as session:
            # Initialize the MCP session and optionally inspect tools
            await session.initialize()

            # Optional: list tools (uncomment to print)
            # tools = await session.list_tools()
            # print(json.dumps(tools.model_dump(), indent=2))

            # Call the built-in discovery tool with empty arguments
            result = await session.call_tool(
                name="discover_agents",
                arguments={},
            )

            # Convert to dict for pretty printing
            print(json.dumps(result.model_dump(), indent=2))
            return result.model_dump()


def main():
    """
    Run the helper-based and raw MCP client discovery examples.
    """
    # asyncio.run(discover_agents_example())
    asyncio.run(raw_mcp_discover_agents_example())


if __name__ == "__main__":
    main()
