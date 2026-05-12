# Search Tools

Search tools give Swarms agents access to current external information. They are useful for research agents, competitive analysis, market monitoring, documentation assistants, and workflows that need source-backed answers. A good search workflow separates discovery, extraction, and synthesis so the final response can cite where each claim came from.

Swarms can use any typed callable as a search tool. Common integrations include Exa, Firecrawl, custom HTTP clients, internal document search, and MCP-based retrieval systems.

## Search Workflow

A reliable search agent usually follows this sequence:

1. Generate focused queries from the user task.
2. Retrieve candidate sources with a search provider.
3. Fetch or extract the most relevant pages.
4. Remove duplicates and low-quality pages.
5. Summarize only from retrieved source text.
6. Return source URLs, dates, and confidence notes.

The model should not blur search results and synthesis. Tool outputs should contain enough metadata for the final answer to explain what was found and what remains uncertain.

## Minimal Search Tool

This example shows a small search wrapper with a structured output. Replace the mocked result with a real provider call.

```python
from datetime import datetime, timezone
from swarms import Agent


def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web for a query and return ranked result metadata."""
    return {
        "query": query,
        "searched_at": datetime.now(timezone.utc).isoformat(),
        "results": [
            {
                "title": "Example source",
                "url": "https://example.com/source",
                "snippet": "A concise summary from the provider.",
                "rank": 1,
            }
        ][:max_results],
    }


agent = Agent(
    agent_name="Source-Aware Research Agent",
    system_prompt=(
        "Use the search tool before answering questions about current events. "
        "Base factual claims on returned URLs and say when evidence is thin."
    ),
    tools=[web_search],
    max_loops=3,
)
```

## Exa Search Pattern

The examples directory includes Exa search agents. A typical production wrapper should:

- Accept a plain query and optional domain filters.
- Limit result count by default.
- Return title, URL, snippet, published date if available, and provider score.
- Fail clearly when `EXA_API_KEY` is missing.

```python
import os


def require_exa_key() -> str:
    key = os.getenv("EXA_API_KEY")
    if not key:
        raise RuntimeError("Set EXA_API_KEY before using Exa search tools")
    return key
```

Use Exa-style search when you need semantic discovery across the public web. For a small known site, a direct crawler or internal index may be more predictable.

## Page Extraction

Search snippets are not enough for high-quality answers. Add a second tool that fetches and extracts page content.

```python
def extract_page(url: str) -> dict:
    """Extract readable content from a URL for source-grounded synthesis."""
    return {
        "url": url,
        "title": "Example page",
        "content": "Clean article or documentation text goes here.",
        "extracted_at": "2026-01-01T00:00:00Z",
    }
```

The agent can call `web_search` to discover URLs and `extract_page` to inspect the strongest sources. This two-step pattern reduces hallucination because the synthesis step sees full source text.

## Query Planning

For difficult research tasks, give the agent an explicit query plan:

```text
Before searching, break the task into 2-4 search questions.
Run one focused query per question.
Prefer primary sources, official documentation, issue threads, papers, and changelogs.
Discard sources that do not directly answer the question.
```

Examples:

| Task | Better query |
| --- | --- |
| "How does this library deploy?" | `site:docs.example.com deployment environment variables` |
| "What changed in the latest version?" | `example project release notes 2026 changelog` |
| "Find implementation examples" | `site:github.com/example-org/example repo \"Agent(\" tools` |

## Source Quality Rules

Not all search results are equal. Prefer:

- Official docs, repositories, changelogs, and standards.
- Maintainer comments in issues or pull requests.
- Primary research papers or dataset pages.
- Recent pages when the topic is time-sensitive.

Avoid relying on:

- SEO copies of documentation.
- Old forum posts for fast-moving libraries.
- Snippets without page extraction.
- Aggregators that do not link to primary sources.

## Output Contract

Search tools should return a predictable shape:

```json
{
  "ok": true,
  "query": "example query",
  "searched_at": "2026-01-01T00:00:00Z",
  "results": [
    {
      "title": "Result title",
      "url": "https://example.com",
      "snippet": "Short source snippet",
      "published_at": null,
      "score": 0.91
    }
  ],
  "warnings": []
}
```

If the search provider times out or returns no results, the final answer should say that the search failed or was inconclusive. Do not let the agent answer as if live evidence was retrieved.

## Related Examples

- `examples/tools/exa_search_agent.py`
- `examples/tools/exa_search_agent_quant.py`
- `examples/guides/web_scraper_agents/web_scraper_agent.py`
- `examples/tools/firecrawl_agents_example.py`

