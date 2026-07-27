# MCP (Model Context Protocol) Examples

[MCP](https://modelcontextprotocol.io) lets an agent pull in tools from an external
server by pointing at a URL. No manual tool wiring — the agent discovers what the
server offers and calls it.

These examples are split by what you're trying to do:

| Folder | Use it when you want to… | Start with |
|---|---|---|
| **[`agents/`](agents/)** | Give an agent tools from an MCP server | [`01_deepwiki_repo_qa.py`](agents/01_deepwiki_repo_qa.py) |
| **[`servers/`](servers/)** | Build an MCP server that agents connect to | [`crypto_price_server.py`](servers/crypto_price_server.py) |
| **[`client/`](client/)** | Call MCP directly with `MCPManager`, without an Agent | [`01_list_tools.py`](client/01_list_tools.py) |

## 30-second version

```python
from swarms import Agent

agent = Agent(
    agent_name="DeepWiki-Agent",
    model_name="gpt-4o-mini",
    mcp_url="https://mcp.deepwiki.com/mcp",   # free, no API key
    max_loops=1,
)

agent.run("What is the swarms framework? Use the deepwiki tools on kyegomez/swarms.")
```

That is the whole integration. Everything in `agents/` is a variation on it —
multiple servers, authentication, mixing MCP tools with local Python tools.

## Which server should I point at?

[`agents/FREE_MCP_SERVERS.md`](agents/FREE_MCP_SERVERS.md) catalogs real, public MCP
servers that work out of the box. Several need no API key at all, so the first four
examples in `agents/` run with nothing but an LLM key.

## Setup

```bash
pip install swarms
export OPENAI_API_KEY="sk-..."       # or any LiteLLM-supported provider
```

Examples that point at `http://localhost:8000/mcp` need a local server first — run one
from [`servers/`](servers/) in a separate terminal.
