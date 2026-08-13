"""
Example 10 — Firecrawl MCP (API key, free tier available).

Firecrawl turns arbitrary web pages into clean markdown an LLM can actually
read: it renders JavaScript, strips navigation and ads, and follows links when
you ask it to crawl rather than scrape a single page.

    Server : https://mcp.firecrawl.dev/{API_KEY}/v2/mcp
    Auth   : API key embedded in the URL path
    Tools  : firecrawl_scrape, firecrawl_crawl, firecrawl_map,
             firecrawl_search, firecrawl_extract

A third auth shape, alongside the two already shown in this folder:

    Example 05 (Exa)     — key as a query parameter  ?exaApiKey=...
    Example 09 (GitHub)  — key as a Bearer header    Authorization: Bearer ...
    Example 10 (this)    — key as a URL path segment /{API_KEY}/

Because the key is part of the URL here, keep it out of logs: build the URL
from the environment at startup and never print the constructed value.

Crawling is the expensive operation — it is billed per page and can walk a
large site quickly. Scrape one page first; crawl only when you mean to.

Get a key (free tier): https://www.firecrawl.dev/

Run:
    export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY, etc.
    export FIRECRAWL_API_KEY=fc-...
    python examples/mcp/agents/10_firecrawl_web_scraping.py
"""

import os
import sys

from swarms import Agent

MODEL = "gpt-5.4"

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

SCRAPER_SYSTEM_PROMPT = (
    "You are a web content analyst. Fetch pages before describing them — "
    "never answer from memory about what a site says, because sites change. "
    "Prefer scraping the specific page that answers the question over "
    "crawling a whole site, since crawling is slow and expensive; crawl only "
    "when the user explicitly wants breadth. Quote the page's own wording for "
    "any factual claim, note the URL each fact came from, and say clearly "
    "when a page failed to load or was empty rather than substituting "
    "assumptions."
)

if __name__ == "__main__" and not FIRECRAWL_API_KEY:
    sys.exit(
        "FIRECRAWL_API_KEY is not set.\n"
        "Get a free-tier key at https://www.firecrawl.dev/ and export it."
    )

agent = Agent(
    agent_name="Firecrawl-Analyst",
    agent_description="Reads and analyzes live web pages via Firecrawl MCP.",
    system_prompt=SCRAPER_SYSTEM_PROMPT,
    model_name=MODEL,
    # The key is a path segment for Firecrawl. Never log this URL.
    mcp_url=f"https://mcp.firecrawl.dev/{FIRECRAWL_API_KEY}/v2/mcp",
    max_loops=2,
)

if __name__ == "__main__":
    result = agent.run(
        "Scrape https://modelcontextprotocol.io/introduction and explain, in "
        "the docs' own terms, what problem MCP solves and what the three "
        "core primitives are. Quote the definitions."
    )
    print(result)
