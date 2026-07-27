"""
Example 6 — Drive agents running on a remote MCP server.

``agent_as_tool_server.py`` exposes a whole swarms ``Agent`` as a single MCP tool
called ``create_agent``. From the client side that is just another tool call, so
``MCPManager`` can spawn and run agents on another process or machine without any
of the raw protocol handling this used to require.

Run the server first:

    python examples/mcp/servers/agent_as_tool_server.py

Then:

    python examples/mcp/client/06_remote_agents.py
"""

import asyncio

from swarms.tools.mcp_manager import MCPManager

SERVER_URL = "http://localhost:8000/mcp"


def one_agent() -> None:
    """Create and run a single remote agent."""
    manager = MCPManager(mcp_url=SERVER_URL)

    result = manager.call_tool(
        "create_agent",
        {
            "agent_name": "Research-Agent",
            "system_prompt": "You are a concise research assistant.",
            "model_name": "gpt-4o-mini",
            "task": "Name the three largest moons of Jupiter.",
        },
    )
    print(f"Research-Agent -> {result}")


async def several_agents() -> None:
    """Run several remote agents concurrently over the same connection."""
    manager = MCPManager(mcp_url=SERVER_URL)

    specs = [
        {
            "agent_name": "Finance-Agent",
            "system_prompt": "You are a financial analyst. Be brief.",
            "model_name": "gpt-4o-mini",
            "task": "What is a P/E ratio?",
        },
        {
            "agent_name": "Science-Agent",
            "system_prompt": "You are a physicist. Be brief.",
            "model_name": "gpt-4o-mini",
            "task": "Why is the sky blue?",
        },
    ]

    results = await asyncio.gather(
        *(manager.acall_tool("create_agent", spec) for spec in specs)
    )

    for spec, result in zip(specs, results):
        print(f"{spec['agent_name']} -> {result}")


if __name__ == "__main__":
    one_agent()
    asyncio.run(several_agents())
