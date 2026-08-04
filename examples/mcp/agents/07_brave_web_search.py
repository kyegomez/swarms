"""
Example 7 — Brave Search MCP (free API key, server runs locally).

Brave publishes an MCP server for its independent search index, but does
*not* host it for you — unlike DeepWiki, Exa, and Firecrawl, there is no
public URL to point at. You run the server yourself and the agent connects
over localhost.

    Server : http://127.0.0.1:8080/mcp   (you start it, see below)
    Auth   : none on the MCP transport — the server holds the Brave key
    Key    : free tier at https://api-dashboard.search.brave.com/app/keys
    Tools  : brave_web_search, brave_news_search, brave_local_search,
             brave_image_search, brave_video_search, brave_summarizer,
             brave_place_search, brave_llm_context

Start the server in a second terminal first:

    export BRAVE_API_KEY=...
    npx -y @brave/brave-search-mcp-server --transport http

It binds to 127.0.0.1:8080 by default (override with BRAVE_MCP_HOST /
BRAVE_MCP_PORT) and serves MCP at /mcp. The HTTP endpoint is
unauthenticated, so keep it on loopback.

Run:
    export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY, etc.
    python 07_brave_web_search.py
"""

from swarms import Agent

# Any LiteLLM model works.
MODEL = "gpt-5.4"

# Matches the server's default bind address; change if you set
# BRAVE_MCP_HOST or BRAVE_MCP_PORT.
BRAVE_MCP_URL = "http://127.0.0.1:8080/mcp"

BRAVE_SYSTEM_PROMPT = (
    "You are a web research specialist who answers questions using Brave "
    "Search. Turn each request into a few precise, differently-worded "
    "queries rather than one broad one, and pick the tool that fits: web "
    "search for general questions, news search when the answer depends on "
    "recent events, local search for places and businesses. Read the "
    "returned results carefully, prefer authoritative primary sources over "
    "aggregators, and corroborate any important claim across more than one "
    "result. Cite every key claim with its URL, include publication dates "
    "when recency matters, separate what the sources establish from your "
    "own inference, and state clearly when the search results do not "
    "answer the question. Never invent facts, quotations, or links."
)

agent = Agent(
    agent_name="Brave-Search-Agent",
    agent_description="Answers questions using Brave Search via a local MCP server.",
    system_prompt=BRAVE_SYSTEM_PROMPT,
    model_name=MODEL,
    mcp_url=BRAVE_MCP_URL,
    max_loops=1,
    reasoning_effort=None,
)

if __name__ == "__main__":
    result = agent.run(
        "Search for the most recent news about open-source multi-agent AI "
        "frameworks and summarize the three most significant items, with "
        "dates and links."
    )
    print(result)
