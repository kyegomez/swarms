"""
Example 4 — Several servers behind one manager.

Pass ``mcp_urls`` and the manager fetches tools from every server, remembers
which server owns which tool, and routes each call accordingly. You never have
to track the mapping yourself.

Run both servers first, in separate terminals:

    python examples/mcp/servers/crypto_price_server.py     # port 8000
    python examples/mcp/servers/okx_crypto_server.py       # port 8001

Then:

    python examples/mcp/client/04_multi_server.py
"""

from swarms.tools.mcp_manager import MCPManager

SERVERS = [
    "http://localhost:8000/mcp",
    "http://localhost:8001/mcp",
]


def main() -> None:
    manager = MCPManager(mcp_urls=SERVERS)

    # One combined view across every configured server.
    print(
        f"tools across {len(SERVERS)} servers: {manager.list_tool_names()}"
    )

    # Calls are routed to whichever server advertised the tool — note these two
    # tools live on different ports, but the call site looks identical.
    print(
        manager.call_tool("get_crypto_price", {"coin_id": "bitcoin"})
    )
    print(
        manager.call_tool(
            "get_okx_crypto_price", {"symbol": "BTC-USDT"}
        )
    )

    # A single LLM response may call tools on different servers at once; the
    # manager fans them out and returns results in order.
    response = {
        "tool_calls": [
            {
                "function": {
                    "name": "get_crypto_price",
                    "arguments": {"coin_id": "bitcoin"},
                }
            },
            {
                "function": {
                    "name": "get_okx_crypto_price",
                    "arguments": {"symbol": "ETH-USDT"},
                }
            },
        ]
    }
    for result in manager.execute_tool_calls(response):
        print(result)

    # Servers can be added later; doing so invalidates the tool cache so the
    # next fetch picks up the new server's tools.
    manager.add_server("http://localhost:8002/mcp")
    print(
        f"after add_server: {len(manager.to_dict()['servers'])} configured"
    )


if __name__ == "__main__":
    main()
