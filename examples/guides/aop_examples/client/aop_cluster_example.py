import json
import asyncio

from swarms.structs.aop import AOPCluster
from swarms.tools.mcp_manager import MCPManager


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

    result = await MCPManager(
        mcp_url="http://localhost:5932/mcp", verbose=False
    ).aexecute_tool_calls(
        tool_call_request,
        output_type="dict",
    )
    print(json.dumps(result, indent=2))
    return result


def main():
    """
    Run the discover_agents_example coroutine.
    """
    asyncio.run(discover_agents_example())


if __name__ == "__main__":
    main()
