"""
Example 6 — Firecrawl MCP (works keyless, better with a free API key).

Firecrawl runs a hosted MCP server that scrapes, crawls, and searches the
web and hands back clean markdown instead of raw HTML. The same URL serves
both modes:

    Server : https://mcp.firecrawl.dev/v2/mcp
    Auth   : none            -> firecrawl_search, firecrawl_scrape,
                                firecrawl_parse, rate-limited per IP
             Bearer API key  -> the full tool surface, including
                                firecrawl_map, firecrawl_crawl,
                                firecrawl_check_crawl_status, firecrawl_extract
    Key    : free tier at https://firecrawl.dev/app/api-keys

``mcp_api_key`` is sent as ``Authorization: Bearer <key>``. Leaving it as
``None`` is not an error — the server simply falls back to the keyless,
rate-limited tool set, so this example runs with nothing but an LLM key.

Run:
    export OPENAI_API_KEY=...             # or ANTHROPIC_API_KEY, etc.
    export FIRECRAWL_API_KEY=...          # optional, unlocks crawl/map/extract
    python 06_firecrawl_web_scraping.py
"""

import os

from swarms import Agent

# Any LiteLLM model works.
MODEL = "gpt-5.4"

# Unset is fine — the agent then talks to Firecrawl in keyless mode.
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

FIRECRAWL_SYSTEM_PROMPT = (
    "You are a web extraction specialist who reads live web pages through "
    "the Firecrawl MCP server. Choose the right tool for the job: search "
    "when you need to find pages, scrape when you already have a URL, map "
    "or crawl when you need to cover a whole site, and extract when the "
    "user wants structured fields. Work only from the content the tools "
    "actually return — never fill gaps from memory or guess at a page's "
    "contents. Quote the exact wording for any figure, price, date, or "
    "claim that matters, attribute every fact to the URL it came from, and "
    "say plainly when a page is unreachable, paywalled, or does not contain "
    "what was asked for."
)

agent = Agent(
    agent_name="Firecrawl-Agent",
    agent_description="Scrapes and searches live web pages via Firecrawl MCP.",
    system_prompt=FIRECRAWL_SYSTEM_PROMPT,
    model_name=MODEL,
    mcp_url="https://mcp.firecrawl.dev/v2/mcp",
    # Sent as "Authorization: Bearer <key>"; None means keyless mode.
    mcp_api_key=FIRECRAWL_API_KEY,
    max_loops=1,
    reasoning_effort=None,
)

if __name__ == "__main__":
    result = agent.run(
        "Scrape https://docs.swarms.world and summarize what the Swarms "
        "framework does, quoting the page's own wording for the main "
        "feature claims."
    )
    print(result)
