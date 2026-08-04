# Agents + MCP

Giving an agent tools from an MCP server. Set `mcp_url` (one server) or `mcp_urls`
(several) and the agent discovers and calls the tools on its own.

## Start here — numbered, in order

Each runs against a real server. The first four need **no MCP API key**; 07 is the
one case where the server is not hosted for you and you start it yourself.

| # | File | Server | Auth |
|---|---|---|---|
| 01 | [`01_deepwiki_repo_qa.py`](01_deepwiki_repo_qa.py) | DeepWiki — Q&A over any public GitHub repo | none |
| 02 | [`02_gitmcp_repo_docs.py`](02_gitmcp_repo_docs.py) | GitMCP — docs/code search for one repo | none |
| 03 | [`03_microsoft_learn_docs.py`](03_microsoft_learn_docs.py) | Microsoft Learn — official Azure/.NET docs | none |
| 04 | [`04_multi_server_agent.py`](04_multi_server_agent.py) | Two servers on one agent | none |
| 05 | [`05_exa_web_search.py`](05_exa_web_search.py) | Exa — web search | free API key |
| 06 | [`06_firecrawl_web_scraping.py`](06_firecrawl_web_scraping.py) | Firecrawl — scrape/crawl pages to markdown | none (key unlocks more tools) |
| 07 | [`07_brave_web_search.py`](07_brave_web_search.py) | Brave Search — web/news/local search | free API key, local server |

See [`FREE_MCP_SERVERS.md`](FREE_MCP_SERVERS.md) for the full catalog of public servers.

## Configuration patterns

| File | Shows |
|---|---|
| [`deepwiki_minimal.py`](deepwiki_minimal.py) | The smallest possible `mcp_url` agent |
| [`mcp_connection_object.py`](mcp_connection_object.py) | `MCPConnection` instead of a bare URL — headers, auth, timeout |
| [`multi_mcp_urls.py`](multi_mcp_urls.py) | `mcp_urls=[...]` for several servers at once |
| [`multi_mcp_walkthrough.py`](multi_mcp_walkthrough.py) | Longer multi-server walkthrough with commentary |
| [`mcp_with_local_tools.py`](mcp_with_local_tools.py) | MCP tools *plus* your own tool schemas on one agent |
| [`tools_list_dictionary.py`](tools_list_dictionary.py) | The raw `tools_list_dictionary` schema format MCP tools are converted into |
| [`finance_agent_mcp.py`](finance_agent_mcp.py) | A realistic finance agent backed by an MCP server |

## Run one

```bash
export OPENAI_API_KEY="sk-..."
python examples/mcp/agents/01_deepwiki_repo_qa.py
```

Examples pointing at `http://localhost:8000/mcp` need a local server — start one from
[`../servers/`](../servers/) first.
