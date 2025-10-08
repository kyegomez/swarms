#!/usr/bin/env python3
"""
Simple example showing how to call the discover_agents tool synchronously.
"""

import json
import asyncio
from swarms.structs.aop import AOPCluster
from swarms.tools.mcp_client_tools import execute_tool_call_simple


def call_discover_agents_sync(server_url="http://localhost:5932/mcp"):
    """
    Synchronously call the discover_agents tool.

    Args:
        server_url: URL of the MCP server

    Returns:
        Dict containing the discovery results
    """

    # Create the tool call request
    tool_call_request = {
        "type": "function",
        "function": {
            "name": "discover_agents",
            "arguments": json.dumps({}),  # Empty = get all agents
        },
    }

    # Run the async function
    return asyncio.run(
        execute_tool_call_simple(
            response=tool_call_request,
            server_path=server_url,
            output_type="dict",
        )
    )


def call_discover_specific_agent_sync(
    agent_name, server_url="http://localhost:5932/mcp"
):
    """
    Synchronously call the discover_agents tool for a specific agent.

    Args:
        agent_name: Name of the specific agent to discover
        server_url: URL of the MCP server

    Returns:
        Dict containing the discovery results
    """

    # Create the tool call request
    tool_call_request = {
        "type": "function",
        "function": {
            "name": "discover_agents",
            "arguments": json.dumps({"agent_name": agent_name}),
        },
    }

    # Run the async function
    return asyncio.run(
        execute_tool_call_simple(
            response=tool_call_request,
            server_path=server_url,
            output_type="dict",
        )
    )


def main():
    """Main function demonstrating discovery tool usage."""

    print("🔍 AOP Agent Discovery Tool Example")
    print("=" * 40)
    print()

    # First, check what tools are available
    print("1. Checking available MCP tools...")
    aop_cluster = AOPCluster(
        urls=["http://localhost:5932/mcp"],
        transport="streamable-http",
    )

    tools = aop_cluster.get_tools(output_type="dict")
    print(f"   Found {len(tools)} tools")

    # Check if discover_agents is available
    discover_tool = aop_cluster.find_tool_by_server_name(
        "discover_agents"
    )
    if not discover_tool:
        print("❌ discover_agents tool not found!")
        print(
            "   Make sure your AOP server is running with agents registered."
        )
        return

    print("✅ discover_agents tool found!")
    print()

    # Discover all agents
    print("2. Discovering all agents...")
    try:
        result = call_discover_agents_sync()

        if isinstance(result, list) and len(result) > 0:
            discovery_data = result[0]

            if discovery_data.get("success"):
                agents = discovery_data.get("agents", [])
                print(f"   ✅ Found {len(agents)} agents:")

                for i, agent in enumerate(agents, 1):
                    print(
                        f"   {i}. {agent.get('agent_name', 'Unknown')}"
                    )
                    print(
                        f"      Role: {agent.get('role', 'worker')}"
                    )
                    print(
                        f"      Description: {agent.get('description', 'No description')}"
                    )
                    print(
                        f"      Tags: {', '.join(agent.get('tags', []))}"
                    )
                    print(
                        f"      Capabilities: {', '.join(agent.get('capabilities', []))}"
                    )
                    print(
                        f"      System Prompt: {agent.get('short_system_prompt', 'No prompt')[:100]}..."
                    )
                    print()
            else:
                print(
                    f"   ❌ Discovery failed: {discovery_data.get('error', 'Unknown error')}"
                )
        else:
            print("   ❌ No valid result returned")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    print()

    # Example of discovering a specific agent (if any exist)
    print("3. Example: Discovering a specific agent...")
    try:
        # Try to discover the first agent specifically
        if isinstance(result, list) and len(result) > 0:
            discovery_data = result[0]
            if discovery_data.get("success") and discovery_data.get(
                "agents"
            ):
                first_agent_name = discovery_data["agents"][0].get(
                    "agent_name"
                )
                if first_agent_name:
                    print(
                        f"   Looking for specific agent: {first_agent_name}"
                    )
                    specific_result = (
                        call_discover_specific_agent_sync(
                            first_agent_name
                        )
                    )

                    if (
                        isinstance(specific_result, list)
                        and len(specific_result) > 0
                    ):
                        specific_data = specific_result[0]
                        if specific_data.get("success"):
                            agent = specific_data.get("agents", [{}])[
                                0
                            ]
                            print(
                                f"   ✅ Found specific agent: {agent.get('agent_name', 'Unknown')}"
                            )
                            print(
                                f"      Model: {agent.get('model_name', 'Unknown')}"
                            )
                            print(
                                f"      Max Loops: {agent.get('max_loops', 1)}"
                            )
                            print(
                                f"      Temperature: {agent.get('temperature', 0.5)}"
                            )
                        else:
                            print(
                                f"   ❌ Specific discovery failed: {specific_data.get('error')}"
                            )
                    else:
                        print("   ❌ No valid specific result")
                else:
                    print(
                        "   ⚠️  No agents found to test specific discovery"
                    )
            else:
                print(
                    "   ⚠️  No agents available for specific discovery"
                )
        else:
            print(
                "   ⚠️  No previous discovery results to use for specific discovery"
            )

    except Exception as e:
        print(f"   ❌ Error in specific discovery: {e}")

    print()
    print("✅ Discovery tool demonstration complete!")
    print()
    print("💡 Usage Summary:")
    print(
        "   • Call discover_agents() with no arguments to get all agents"
    )
    print(
        "   • Call discover_agents(agent_name='AgentName') to get specific agent"
    )
    print(
        "   • Each agent returns: name, description, role, tags, capabilities, system prompt, etc."
    )


if __name__ == "__main__":
    main()
