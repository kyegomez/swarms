#!/usr/bin/env python3
"""
Example demonstrating the new agent information tools in AOP.

This example shows how to use the new MCP tools for getting agent information.
"""

import json
import asyncio
from swarms.structs.aop import AOPCluster
from swarms.tools.mcp_client_tools import execute_tool_call_simple


async def demonstrate_new_agent_tools():
    """Demonstrate the new agent information tools."""

    # Create AOP cluster connection
    AOPCluster(
        urls=["http://localhost:5932/mcp"],
        transport="streamable-http",
    )

    print("🔧 New AOP Agent Information Tools Demo")
    print("=" * 50)
    print()

    # 1. List all agents
    print("1. Listing all agents...")
    try:
        tool_call = {
            "type": "function",
            "function": {"name": "list_agents", "arguments": "{}"},
        }

        result = await execute_tool_call_simple(
            response=tool_call,
            server_path="http://localhost:5932/mcp",
            output_type="dict",
            verbose=False,
        )

        if isinstance(result, list) and len(result) > 0:
            data = result[0]
            if data.get("success"):
                agent_names = data.get("agent_names", [])
                print(
                    f"   Found {len(agent_names)} agents: {agent_names}"
                )
            else:
                print(f"   Error: {data.get('error')}")
        else:
            print("   No valid result returned")
    except Exception as e:
        print(f"   Error: {e}")
    print()

    # 2. Get details for a specific agent
    print("2. Getting details for a specific agent...")
    try:
        tool_call = {
            "type": "function",
            "function": {
                "name": "get_agent_details",
                "arguments": json.dumps(
                    {"agent_name": "Research-Agent"}
                ),
            },
        }

        result = await execute_tool_call_simple(
            response=tool_call,
            server_path="http://localhost:5932/mcp",
            output_type="dict",
            verbose=False,
        )

        if isinstance(result, list) and len(result) > 0:
            data = result[0]
            if data.get("success"):
                data.get("agent_info", {})
                discovery_info = data.get("discovery_info", {})
                print(
                    f"   Agent: {discovery_info.get('agent_name', 'Unknown')}"
                )
                print(
                    f"   Description: {discovery_info.get('description', 'No description')}"
                )
                print(
                    f"   Model: {discovery_info.get('model_name', 'Unknown')}"
                )
                print(f"   Tags: {discovery_info.get('tags', [])}")
                print(
                    f"   Capabilities: {discovery_info.get('capabilities', [])}"
                )
            else:
                print(f"   Error: {data.get('error')}")
        else:
            print("   No valid result returned")
    except Exception as e:
        print(f"   Error: {e}")
    print()

    # 3. Get info for multiple agents
    print("3. Getting info for multiple agents...")
    try:
        tool_call = {
            "type": "function",
            "function": {
                "name": "get_agents_info",
                "arguments": json.dumps(
                    {
                        "agent_names": [
                            "Research-Agent",
                            "DataAnalyst",
                            "Writer",
                        ]
                    }
                ),
            },
        }

        result = await execute_tool_call_simple(
            response=tool_call,
            server_path="http://localhost:5932/mcp",
            output_type="dict",
            verbose=False,
        )

        if isinstance(result, list) and len(result) > 0:
            data = result[0]
            if data.get("success"):
                agents_info = data.get("agents_info", [])
                not_found = data.get("not_found", [])
                print(
                    f"   Found {len(agents_info)} agents out of {data.get('total_requested', 0)} requested"
                )
                for agent in agents_info:
                    discovery_info = agent.get("discovery_info", {})
                    print(
                        f"   • {discovery_info.get('agent_name', 'Unknown')}: {discovery_info.get('description', 'No description')}"
                    )
                if not_found:
                    print(f"   Not found: {not_found}")
            else:
                print(f"   Error: {data.get('error')}")
        else:
            print("   No valid result returned")
    except Exception as e:
        print(f"   Error: {e}")
    print()

    # 4. Search for agents
    print("4. Searching for agents...")
    try:
        tool_call = {
            "type": "function",
            "function": {
                "name": "search_agents",
                "arguments": json.dumps(
                    {
                        "query": "data",
                        "search_fields": [
                            "name",
                            "description",
                            "tags",
                            "capabilities",
                        ],
                    }
                ),
            },
        }

        result = await execute_tool_call_simple(
            response=tool_call,
            server_path="http://localhost:5932/mcp",
            output_type="dict",
            verbose=False,
        )

        if isinstance(result, list) and len(result) > 0:
            data = result[0]
            if data.get("success"):
                matching_agents = data.get("matching_agents", [])
                print(
                    f"   Found {len(matching_agents)} agents matching 'data'"
                )
                for agent in matching_agents:
                    print(
                        f"   • {agent.get('agent_name', 'Unknown')}: {agent.get('description', 'No description')}"
                    )
                    print(f"     Tags: {agent.get('tags', [])}")
                    print(
                        f"     Capabilities: {agent.get('capabilities', [])}"
                    )
            else:
                print(f"   Error: {data.get('error')}")
        else:
            print("   No valid result returned")
    except Exception as e:
        print(f"   Error: {e}")
    print()

    print("✅ New agent tools demonstration complete!")
    print()
    print("💡 Available Tools:")
    print(
        "   • discover_agents - Get discovery info for all or specific agents"
    )
    print(
        "   • get_agent_details - Get detailed info for a single agent"
    )
    print(
        "   • get_agents_info - Get detailed info for multiple agents"
    )
    print("   • list_agents - Get simple list of all agent names")
    print("   • search_agents - Search agents by keywords")


def main():
    """Main function to run the demonstration."""
    asyncio.run(demonstrate_new_agent_tools())


if __name__ == "__main__":
    main()
