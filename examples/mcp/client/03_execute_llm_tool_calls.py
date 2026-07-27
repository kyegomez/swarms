"""
Example 3 — Execute the tool calls an LLM asked for.

When a model replies with tool calls, hand that response straight to
``execute_tool_calls()``. The manager parses the calls, routes each one to the
server that advertised the tool, and returns the results in order. This is the
step an ``Agent`` performs for you between LLM turns.

Run a server first:

    python examples/mcp/servers/crypto_price_server.py

Then:

    python examples/mcp/client/03_execute_llm_tool_calls.py
"""

import asyncio

from swarms.tools.mcp_manager import MCPManager

SERVER_URL = "http://localhost:8000/mcp"

# The shape an LLM produces when it decides to call a tool.
LLM_RESPONSE = {
    "function": {
        "name": "get_crypto_price",
        "arguments": {"coin_id": "bitcoin"},
    }
}


def main() -> None:
    manager = MCPManager(mcp_url=SERVER_URL)

    # dict (default) — a list of result dicts.
    results = manager.execute_tool_calls(LLM_RESPONSE)
    print(f"dict:\n{results}\n")

    # json — one JSON string, handy for logging or feeding back to the model.
    as_json = manager.execute_tool_calls(
        LLM_RESPONSE, output_type="json"
    )
    print(f"json:\n{as_json}\n")

    # str — plain text, for dropping into a prompt.
    as_text = manager.execute_tool_calls(
        LLM_RESPONSE, output_type="str"
    )
    print(f"str:\n{as_text}\n")

    # Already have results and just want them rendered differently?
    print(
        f"re-formatted: {MCPManager.format_results(results, 'str')}"
    )


async def main_async() -> None:
    manager = MCPManager(mcp_url=SERVER_URL)
    results = await manager.aexecute_tool_calls(LLM_RESPONSE)
    print(f"async: {results}")


if __name__ == "__main__":
    main()
    asyncio.run(main_async())
