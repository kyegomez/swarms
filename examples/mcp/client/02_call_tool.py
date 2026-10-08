"""
Example 2 — Call a single tool by name.

``call_tool(name, arguments)`` is the direct route: no LLM, no agent, just invoke
a tool on the server. Useful for testing a server, or for wiring MCP tools into
code that decides for itself what to call.

Run a server first:

    python examples/mcp/servers/crypto_price_server.py

Then:

    python examples/mcp/client/02_call_tool.py
"""

import asyncio

from swarms.tools.mcp_manager import MCPManager

SERVER_URL = "http://localhost:8000/mcp"


def sync_call() -> None:
    """The synchronous form — fine from ordinary code."""
    manager = MCPManager(mcp_url=SERVER_URL)

    result = manager.call_tool(
        "get_crypto_price", {"coin_id": "bitcoin"}
    )
    print(f"sync result: {result}")


async def async_call() -> None:
    """The async form — use inside an event loop."""
    manager = MCPManager(mcp_url=SERVER_URL)

    result = await manager.acall_tool(
        "get_crypto_price", {"coin_id": "ethereum"}
    )
    print(f"async result: {result}")


if __name__ == "__main__":
    sync_call()
    asyncio.run(async_call())
