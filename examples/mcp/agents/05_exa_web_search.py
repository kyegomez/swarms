"""
Example 5 — Exa MCP (free tier, needs a free API key).

Exa provides a hosted MCP server for high-quality web search and content
retrieval. Unlike the earlier examples, it requires an API key — but Exa's
key is free to obtain and has a free usage tier.

    Server : https://mcp.exa.ai/mcp
    Auth   : Exa API key (free at https://dashboard.exa.ai/api-keys)
    Tools  : web_search_exa, get_contents, find_similar, ...

Exa's remote MCP takes the key as the ``exaApiKey`` query parameter, so we
build the URL from the ``EXA_API_KEY`` environment variable. (For servers
that expect a bearer token instead, the Agent also accepts
``mcp_api_key="env:VAR_NAME"``, which sends ``Authorization: Bearer <key>``.)

Run:
    export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY, etc.
    export EXA_API_KEY=...           # free from https://dashboard.exa.ai
    python 5_exa_web_search.py
"""

import os

from swarms import Agent

# Any LiteLLM model works; gpt-4o-mini is cheap and good at tool use.
MODEL = "gpt-5.4"

EXA_API_KEY = os.getenv("EXA_API_KEY")

WEB_SEARCH_SYSTEM_PROMPT = (
    "You are a web research specialist who answers questions by searching "
    "the live web with Exa. Translate each request into precise search "
    "queries, inspect the most relevant and recent sources, and synthesize "
    "their findings into a direct, well-organized response. Prioritize "
    "authoritative primary sources, verify important claims across sources "
    "when possible, distinguish facts from uncertainty, include publication "
    "dates when recency matters, and cite every key claim with a working "
    "source link. Never invent facts, quotations, or URLs; if reliable "
    "evidence cannot be found, state that clearly."
)

agent = Agent(
    agent_name="Exa-Search-Agent",
    agent_description="Answers questions using live web search via Exa MCP.",
    system_prompt=WEB_SEARCH_SYSTEM_PROMPT,
    model_name=MODEL,
    # Exa authenticates via the exaApiKey query parameter.
    mcp_url=f"https://mcp.exa.ai/mcp?exaApiKey={EXA_API_KEY}",
    max_loops=1,
    reasoning_effort=None,
)

if __name__ == "__main__":
    result = agent.run(
        "Use Exa web search to find the three most recent notable "
        "developments in open-source multi-agent AI frameworks, with links."
    )
    print(result)
