# Search Tools

Search tools give Swarms agents a controlled way to retrieve external documents, web results, internal knowledge base entries, or indexed records. They are best used when the agent needs fresh context, citations, or a narrow set of retrieved passages before answering.

## Common Use Cases

- Retrieving web search results for a research agent
- Querying an internal document index
- Looking up product, policy, or support articles
- Fetching recent news or public reports
- Combining search results with a summarization or review agent

## Example

```python
import os
import requests


def web_search(query: str, limit: int = 5) -> str:
    """Search a provider API and return serialized results."""
    api_key = os.getenv("SEARCH_API_KEY")
    response = requests.get(
        "https://api.example.com/search",
        params={"q": query, "limit": limit, "api_key": api_key},
        timeout=20,
    )
    response.raise_for_status()
    return response.text
```

```python
from swarms import Agent

research_agent = Agent(
    agent_name="Search-Agent",
    model_name="gpt-4.1",
    tools=[web_search],
)

research_agent.run("Find recent sources about multimodal agent evaluation.")
```

## Retrieval Guidelines

- Keep result counts small enough for the agent to inspect.
- Return titles, URLs, snippets, and timestamps when available.
- Preserve source URLs so the agent can cite or compare evidence.
- Handle empty results with a clear message instead of raising opaque errors.
- Use provider filters for domain, date range, or document type when possible.

For lower-level tool registration patterns, see the [tool system guide](../swarms/tools/main.md).
