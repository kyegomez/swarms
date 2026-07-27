# MCP Servers

Servers built with `FastMCP` that expose tools for agents to call. Run one, then
point an agent at its URL.

| File | What it exposes | Port |
|---|---|---|
| [`crypto_price_server.py`](crypto_price_server.py) | `get_crypto_price` — live coin prices | 8000 |
| [`okx_crypto_server.py`](okx_crypto_server.py) | `get_okx_crypto_price` — prices from OKX | 8001 |
| [`agent_as_tool_server.py`](agent_as_tool_server.py) | `create_agent` — wraps a whole swarms Agent as one MCP tool | 8000 |
| [`streamable_http_server.py`](streamable_http_server.py) | Stateful vs. stateless streamable-HTTP transport config | 8000 |

## Run one, connect an agent

Terminal 1 — start the server:

```bash
python examples/mcp/servers/crypto_price_server.py
```

Terminal 2 — point an agent at it:

```python
from swarms import Agent

agent = Agent(
    agent_name="Crypto-Agent",
    model_name="gpt-4o-mini",
    mcp_url="http://localhost:8000/mcp",
    max_loops=1,
)
agent.run("What is the current price of Bitcoin?")
```

Most examples under [`../agents/`](../agents/) and [`../client/`](../client/) that use
`http://localhost:8000/mcp` expect `crypto_price_server.py` to be running.

## Agents as tools

[`agent_as_tool_server.py`](agent_as_tool_server.py) is the interesting one: it turns a
swarms `Agent` into an MCP tool, so *another* agent — or any MCP client — can spawn and
run it remotely. That's how you compose swarms across process or machine boundaries.
