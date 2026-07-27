# Free MCP Servers for Swarms Agents

A catalog of real, public [MCP](https://modelcontextprotocol.io) servers you can
plug straight into the Swarms `Agent` class — most require **no authentication**,
a few use a **free-tier API key**. Runnable examples live in
this folder, numbered `01`–`05`.

> Availability and URLs change over time — verify a server before depending on it.
> Last reviewed against known-public servers as of early 2026.

---

## Using an MCP server with an Agent

The `Agent` class connects to MCP servers over HTTP. Point it at a URL and it
fetches that server's tools on startup; the model calls them as needed.

```python
from swarms import Agent

agent = Agent(
    agent_name="MCP-Agent",
    model_name="gpt-4o-mini",              # any LiteLLM model with tool use
    mcp_url="https://mcp.deepwiki.com/mcp",  # one server
    max_loops=1,
)
print(agent.run("Use your tools to explain the kyegomez/swarms repo."))
```

**Several servers at once** — pass a list; the agent gets the union of their tools:

```python
agent = Agent(
    model_name="gpt-4o-mini",
    mcp_urls=[
        "https://mcp.deepwiki.com/mcp",
        "https://learn.microsoft.com/api/mcp",
    ],
    max_loops=2,
)
```

**Transport** is auto-detected: any `https://` URL uses streamable-HTTP. A few
servers are SSE-only (endpoint ends in `/sse`) — most of those also expose a
streamable-HTTP endpoint; prefer it when available.

---

## Free — no authentication (works out of the box)

| Server | URL | What it does |
|---|---|---|
| **DeepWiki** | `https://mcp.deepwiki.com/mcp` | Q&A over any public GitHub repo's docs |
| **GitMCP** | `https://gitmcp.io/<owner>/<repo>` | Turns a single repo into a docs/code MCP server |
| **Microsoft Learn** | `https://learn.microsoft.com/api/mcp` | Official Microsoft / Azure / .NET documentation |
| **AWS Knowledge** | `https://knowledge-mcp.global.api.aws` | Official AWS docs, API references, what's-new |
| **Cloudflare Docs** | `https://docs.mcp.cloudflare.com/mcp` | Cloudflare product documentation |
| **Hugging Face** | `https://huggingface.co/mcp` | Search models / datasets / Spaces (optional HF token unlocks more) |
| **Context7** | `https://mcp.context7.com/mcp` | Up-to-date docs for thousands of libraries (rate-limited without a free key) |
| **Globalping** | `https://mcp.globalping.dev/sse` | Run ping / traceroute / DNS / MTR from a global probe network |
| **Semgrep** | `https://mcp.semgrep.ai/mcp` | Static-analysis security scanning of code |

```python
# Example: repo-scoped documentation assistant via GitMCP
agent = Agent(
    model_name="gpt-4o-mini",
    mcp_url="https://gitmcp.io/kyegomez/swarms",
    max_loops=1,
)
```

---

## Free tier — needs a free API key

These are free to use but require a key. How the key is passed differs per server.

| Server | Endpoint / auth | What it does |
|---|---|---|
| **Exa** | `https://mcp.exa.ai/mcp?exaApiKey=…` (query param) | Web search + content retrieval |
| **Tavily** | `https://mcp.tavily.com/mcp/?tavilyApiKey=…` (query param) | Web search built for agents |
| **Firecrawl** | hosted MCP + `FIRECRAWL_API_KEY` | Scrape/crawl sites into clean markdown |
| **Ref** (ref.tools) | hosted + key | Fast documentation search across frameworks |
| **Brave Search** | hosted/local + key | Web + local search |
| **Apify / Bright Data** | hosted + key | Web-scraping / data-extraction actors |

**Key as a query parameter** (Exa, Tavily):

```python
import os
key = os.environ["EXA_API_KEY"]  # free at https://dashboard.exa.ai/api-keys
agent = Agent(
    model_name="gpt-4o-mini",
    mcp_url=f"https://mcp.exa.ai/mcp?exaApiKey={key}",
    max_loops=1,
)
```

**Key as a Bearer token** — the Agent sends `Authorization: Bearer <key>`; use
`env:VAR` to keep the secret out of source:

```python
agent = Agent(
    model_name="gpt-4o-mini",
    mcp_url="https://example-server.com/mcp",
    mcp_api_key="env:MY_SERVER_KEY",
)
```

**Key in a custom header:**

```python
agent = Agent(
    model_name="gpt-4o-mini",
    mcp_url="https://example-server.com/mcp",
    mcp_headers={"x-api-key": os.environ["MY_SERVER_KEY"]},
)
```

---

## OAuth — free with your own account (not no-auth)

These authenticate to *your* account and give the agent real actions in that
service. Provide a token via `mcp_api_key="env:VAR"` (Bearer) or `mcp_headers`.

`GitHub` (`https://api.githubcopilot.com/mcp/`), `Notion`, `Linear`, `Asana`,
`Atlassian`, `Sentry`, `Stripe`, `PayPal`, `Vercel`, `Cloudflare` (bindings),
and many more.

```python
agent = Agent(
    model_name="gpt-4o-mini",
    mcp_url="https://api.githubcopilot.com/mcp/",
    mcp_api_key="env:GITHUB_TOKEN",
)
```

---

## Where to discover more

Registries that list hundreds of servers with their transports and auth:

- **PulseMCP** — https://www.pulsemcp.com
- **Glama** — https://glama.ai/mcp/servers
- **Smithery** — https://smithery.ai
- **mcp.run** — https://www.mcp.run

---

## Model note

Some current Anthropic models (e.g. `claude-sonnet-5`, `claude-opus-4-8`) can
fail through the LiteLLM wrapper with:

```
AnthropicException: "thinking.type.enabled" is not supported for this model.
Use "thinking.type.adaptive" and "output_config.effort".
```

Until that's fixed in the wrapper, the examples use `gpt-4o-mini`, which works
cleanly for MCP tool use. Swap `MODEL` for any LiteLLM-supported model you have a
key for.

---

## Runnable examples

The numbered examples in this folder:

| File | Server | Auth |
|---|---|---|
| [`01_deepwiki_repo_qa.py`](01_deepwiki_repo_qa.py) | DeepWiki | none |
| [`02_gitmcp_repo_docs.py`](02_gitmcp_repo_docs.py) | GitMCP | none |
| [`03_microsoft_learn_docs.py`](03_microsoft_learn_docs.py) | Microsoft Learn | none |
| [`04_multi_server_agent.py`](04_multi_server_agent.py) | DeepWiki + Microsoft Learn | none |
| [`05_exa_web_search.py`](05_exa_web_search.py) | Exa | free API key |

```bash
export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY, etc.
python examples/mcp/agents/01_deepwiki_repo_qa.py
```
