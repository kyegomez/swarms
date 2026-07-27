"""
Example 1 — Discover the tools a server offers.

``MCPManager`` is the single entry point for MCP in swarms. Point it at a server
and ``get_tools()`` returns OpenAI function-calling schemas, ready to hand to an
LLM. This is exactly what an ``Agent`` does internally when you set ``mcp_url``.

Run a server first:

    python examples/mcp/servers/crypto_price_server.py

Then:

    python examples/mcp/client/01_list_tools.py
"""

import json

from swarms.tools.mcp_manager import MCPManager

SERVER_URL = "http://localhost:8000/mcp"


def main() -> None:
    manager = MCPManager(mcp_url=SERVER_URL)

    print(f"configured: {manager.enabled}")

    # Just the names — the cheapest way to see what a server can do.
    print(f"tool names: {manager.list_tool_names()}")

    # Full OpenAI function-calling schemas (the default format).
    tools = manager.get_tools()
    print(f"\n{len(tools)} tool schema(s):")
    print(json.dumps(tools, indent=2)[:800])

    # Raw MCP schemas instead, if you want the protocol's own shape.
    mcp_tools = manager.get_tools(format="mcp")
    print(f"\nraw MCP format: {len(mcp_tools)} tool(s)")

    # Schemas are cached after the first fetch. Force a re-fetch when the
    # server's tools may have changed.
    manager.get_tools(force_refresh=True)
    print("re-fetched after force_refresh=True")


if __name__ == "__main__":
    main()
