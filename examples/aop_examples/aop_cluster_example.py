import json
import asyncio

from swarms.structs.aop import AOPCluster
from swarms.tools.mcp_client_tools import execute_tool_call_simple


async def discover_agents_example():
    """Example of how to call the discover_agents tool."""

    # Create AOP cluster connection
    aop_cluster = AOPCluster(
        urls=["http://localhost:5932/mcp"],
        transport="streamable-http",
    )

    # Check if discover_agents tool is available
    discover_tool = aop_cluster.find_tool_by_server_name(
        "discover_agents"
    )
    if discover_tool:
        try:
            # Create the tool call request
            tool_call_request = {
                "type": "function",
                "function": {
                    "name": "discover_agents",
                    "arguments": json.dumps(
                        {}
                    ),  # No specific agent name = get all
                },
            }

            # Execute the tool call
            result = await execute_tool_call_simple(
                response=tool_call_request,
                server_path="http://localhost:5932/mcp",
                output_type="dict",
                verbose=False,
            )

            print(json.dumps(result, indent=2))

            # Parse the result
            if isinstance(result, list) and len(result) > 0:
                discovery_data = result[0]
                if discovery_data.get("success"):
                    agents = discovery_data.get("agents", [])
                    return agents
                else:
                    return None
            else:
                return None

        except Exception:
            return None
    else:
        return None


def main():
    """Main function to run the discovery example."""
    # Run the async function
    return asyncio.run(discover_agents_example())


if __name__ == "__main__":
    main()
