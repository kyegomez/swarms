# MCP Client (no Agent)

Talking to MCP servers directly through **`MCPManager`** — the same class an
`Agent` uses internally. Point it at one or more servers and it handles transport
selection, authentication, tool discovery, caching, and routing each call to the
server that owns the tool.

Useful for inspecting a server, testing tools, or building your own layer on top
of MCP without an agent in the loop.

```python
from swarms.tools.mcp_manager import MCPManager

manager = MCPManager(mcp_url="http://localhost:8000/mcp")

manager.list_tool_names()                                  # what's available
manager.get_tools()                                        # OpenAI schemas for an LLM
manager.call_tool("get_crypto_price", {"coin_id": "btc"})  # call one directly
manager.execute_tool_calls(llm_response)                   # run what a model asked for
```

## Examples, in order

| # | File | Shows |
|---|---|---|
| 01 | [`01_list_tools.py`](01_list_tools.py) | Discover tools: `list_tool_names()`, `get_tools()`, `openai` vs `mcp` format, cache and `force_refresh` |
| 02 | [`02_call_tool.py`](02_call_tool.py) | Call one tool by name — `call_tool()` and `acall_tool()` |
| 03 | [`03_execute_llm_tool_calls.py`](03_execute_llm_tool_calls.py) | Run the tool calls in an LLM response; `dict` / `json` / `str` output |
| 04 | [`04_multi_server.py`](04_multi_server.py) | Several servers on one manager, automatic routing, `add_server()` |
| 05 | [`05_auth_and_config.py`](05_auth_and_config.py) | API keys, bearer tokens, headers, `env:` secrets, OAuth, per-server auth |
| 06 | [`06_remote_agents.py`](06_remote_agents.py) | Spawn and run agents on a remote MCP server |

## Running them

Most expect a local server on `http://localhost:8000/mcp`:

```bash
python examples/mcp/servers/crypto_price_server.py     # terminal 1
python examples/mcp/client/01_list_tools.py            # terminal 2
```

[`04_multi_server.py`](04_multi_server.py) also wants `okx_crypto_server.py` on port
8001, and [`06_remote_agents.py`](06_remote_agents.py) wants `agent_as_tool_server.py`.
[`05_auth_and_config.py`](05_auth_and_config.py) makes no connections at all — it just
prints how each configuration is interpreted.

## Sync or async

Every operation has both forms: `get_tools` / `aget_tools`, `call_tool` /
`acall_tool`, `execute_tool_calls` / `aexecute_tool_calls`. The synchronous ones are
safe to call from ordinary code, including from inside a running event loop.
