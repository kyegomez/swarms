# MCP Client Reference

`MCPManager` is the single entry point for MCP in swarms. Point it at one or more servers and it handles transport selection, authentication, tool discovery, caching, and routing each tool call to the server that owns it. An `Agent` uses this class internally whenever you set `mcp_url` or `mcp_urls`.

> **Migration note.** The standalone functions in `swarms.tools.mcp_client_tools` — `aget_mcp_tools`, `get_mcp_tools_sync`, `get_tools_for_multiple_mcp_servers`, `execute_tool_call_simple`, and `execute_multiple_tools_on_multiple_mcp_servers` — have been removed. See [Migrating from mcp_client_tools](#migrating-from-mcp_client_tools) for direct replacements.

## Table of Contents

- [Construction](#construction)
- [get_tools](#get_tools)
- [list_tool_names](#list_tool_names)
- [call_tool](#call_tool)
- [execute_tool_calls](#execute_tool_calls)
- [Authentication](#authentication)
- [Managing servers and caches](#managing-servers-and-caches)
- [Migrating from mcp_client_tools](#migrating-from-mcp_client_tools)

## Construction

```python
from swarms.tools.mcp_manager import MCPManager

# One server
manager = MCPManager(mcp_url="http://localhost:8000/mcp")

# Several servers
manager = MCPManager(mcp_urls=[
    "http://localhost:8000/mcp",
    "http://localhost:8001/mcp",
])
```

### Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `mcp_url` | `str \| MCPConnection \| Dict` | No | A single server: URL, connection object, or dict |
| `mcp_urls` | `List[str \| MCPConnection \| Dict]` | No | Several servers; entries may mix forms |
| `mcp_config` | `MCPConnection \| Dict` | No | Full configuration for one server |
| `mcp_configs` | `List[MCPConnection \| Dict]` | No | Full configuration for several servers |
| `api_key` | `str` | No | API key applied to servers that do not define their own |
| `authorization_token` | `str` | No | Bearer token applied to servers that do not define their own |
| `oauth` | `MCPOAuthConfig \| Dict` | No | OAuth 2.1 configuration |
| `headers` | `Dict[str, str]` | No | Extra headers merged into every request |
| `transport` | `str` | No | Force `"streamable_http"`, `"sse"`, or `"stdio"`; auto-detected otherwise |
| `timeout` | `int` | No | Request timeout in seconds (default: 30) |
| `agent_name` | `str` | No | Name used in log messages (default: `"agent"`) |
| `verbose` | `bool` | No | Verbose logging (default: `False`) |
| `retry_attempts` | `int` | No | Retries per operation (default: 3) |

Hyphenated transports (`"streamable-http"`) are normalized automatically.

## get_tools

Fetch tools from every configured server.

```python
tools = manager.get_tools()                      # OpenAI function-calling schemas
raw = manager.get_tools(format="mcp")            # raw MCP schemas
fresh = manager.get_tools(force_refresh=True)    # bypass the cache
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `format` | `"openai" \| "mcp"` | No | Schema shape (default: `"openai"`) |
| `force_refresh` | `bool` | No | Re-fetch instead of using cached schemas |

**Returns** `List[Dict[str, Any]]`. Async form: `await manager.aget_tools(...)`.

## list_tool_names

```python
manager.list_tool_names()
# ['get_crypto_price', 'get_okx_crypto_price']
```

**Returns** `List[str]` — every tool name across all configured servers.

## call_tool

Call one tool directly, without an LLM.

```python
result = manager.call_tool("get_crypto_price", {"coin_id": "bitcoin"})
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `name` | `str` | Yes | Tool name |
| `arguments` | `Dict[str, Any]` | No | Tool arguments |

**Returns** a result envelope:

```python
{
    "tool": "get_crypto_price",
    "server": "http://localhost:8000/mcp",
    "arguments": {"coin_id": "bitcoin"},
    "is_error": False,
    "result": "Current price of Bitcoin: $64,601.00",
}
```

Async form: `await manager.acall_tool(...)`.

## execute_tool_calls

Run the tool calls contained in an LLM response. Each call is routed to the server that advertised the tool, and results come back in the order the calls appeared.

```python
response = {
    "function": {
        "name": "get_crypto_price",
        "arguments": {"coin_id": "bitcoin"},
    }
}

results = manager.execute_tool_calls(response)                     # list of dicts
as_json = manager.execute_tool_calls(response, output_type="json") # JSON string
as_text = manager.execute_tool_calls(response, output_type="str")  # plain text
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `response` | `Any` | Yes | An LLM response, a single call dict, or a list of calls |
| `output_type` | `"dict" \| "json" \| "str"` | No | Result format (default: `"dict"`) |

**Returns** `List[Dict[str, Any]]`, or a string for `"json"` / `"str"`.

Async form: `await manager.aexecute_tool_calls(...)`. To re-render results you already have, use the static `MCPManager.format_results(results, output_type)`.

When a tool returns structured data, its payload arrives as a JSON string in the `result` field:

```python
import json

payload = json.loads(results[0]["result"])
```

## Authentication

```python
from swarms.schemas.mcp_schemas import MCPConnection, MCPOAuthConfig

# API key
MCPManager(mcp_url="https://api.example.com/mcp", api_key="sk-...")

# Bearer token
MCPManager(mcp_url="https://api.example.com/mcp", authorization_token="ey...")

# Full control, including custom headers and timeouts
MCPManager(mcp_config=MCPConnection(
    url="https://api.example.com/mcp",
    headers={"X-Tenant": "acme"},
    transport="streamable_http",
    timeout=20,
))

# Secrets read from the environment at connection time
MCPConnection(url="https://api.example.com/mcp", api_key="env:EXAMPLE_MCP_KEY")

# OAuth 2.1
MCPConnection(url="https://api.example.com/mcp", oauth=MCPOAuthConfig(
    grant_type="client_credentials",
    client_id="example-client",
    client_secret="env:EXAMPLE_CLIENT_SECRET",
    token_url="https://api.example.com/oauth/token",
))
```

Different servers may use different authentication on the same manager — pass a mix of URLs and `MCPConnection` objects to `mcp_urls`.

## Managing servers and caches

| Method | Description |
|---|---|
| `enabled` | Property — `True` when at least one server is configured |
| `add_server(server)` | Register another server and invalidate the tool cache |
| `clear_cache()` | Drop cached tool schemas and routing information |
| `clear_auth_cache()` | Forget in-process OAuth providers and cached tokens |
| `to_dict()` | Serializable, secret-redacted view of the configuration |

## Error handling

Failures raise the agent MCP exceptions from `swarms.schemas.agent_mcp_errors`:

| Exception | Raised when |
|---|---|
| `AgentMCPConnectionError` | The server cannot be reached, or authentication fails |
| `AgentMCPToolError` | A tool call fails on the server |
| `AgentMCPError` | Base class for both |

Operations retry with backoff up to `retry_attempts` times before raising.

## Migrating from mcp_client_tools

| Removed function | Replacement |
|---|---|
| `aget_mcp_tools(server_path=URL)` | `await MCPManager(mcp_url=URL).aget_tools()` |
| `get_mcp_tools_sync(server_path=URL)` | `MCPManager(mcp_url=URL).get_tools()` |
| `get_tools_for_multiple_mcp_servers(urls=URLS)` | `MCPManager(mcp_urls=URLS).get_tools()` |
| `execute_tool_call_simple(response=R, server_path=URL)` | `await MCPManager(mcp_url=URL).aexecute_tool_calls(R)` |
| `execute_multiple_tools_on_multiple_mcp_servers(responses=R, urls=URLS)` | `await MCPManager(mcp_urls=URLS).aexecute_tool_calls(R)` |
| `MCPError`, `MCPConnectionError`, `MCPToolError`, `MCPExecutionError` | `AgentMCPError`, `AgentMCPConnectionError`, `AgentMCPToolError` |

Two behavioral differences worth noting:

1. **`output_type` on tool fetching is gone.** `get_tools_for_multiple_mcp_servers` accepted the argument but never applied it. Use `format="openai"` or `format="mcp"` to choose the schema shape.
2. **Execution results are wrapped.** The old functions returned the raw MCP `CallToolResult` dump. `execute_tool_calls` returns one envelope per call — `{"tool", "server", "arguments", "is_error", "result"}` — so results from several servers stay attributable. Read `result["result"]` for the tool's own payload.

## Best practices

1. Build one `MCPManager` and reuse it — tool schemas are cached per instance.
2. Prefer `mcp_urls` over several managers when an agent needs multiple servers; routing is then automatic.
3. Keep secrets out of source with `env:NAME` indirection in `MCPConnection`.
4. Use the async methods inside an event loop; the sync ones are safe everywhere else, including from within a running loop.
5. Check `is_error` on each result envelope rather than assuming success.

## Related

- [Examples](https://github.com/kyegomez/swarms/tree/master/examples/mcp) — runnable agent, server, and client examples
- [MCP schemas](https://github.com/kyegomez/swarms/blob/master/swarms/schemas/mcp_schemas.py) — `MCPConnection`, `MCPOAuthConfig`
