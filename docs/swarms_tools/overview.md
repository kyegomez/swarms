# Swarms Tools

Swarms tools are callable functions that agents can use to reach outside the model, fetch data, call APIs, run deterministic logic, or hand off work to external systems. A tool can be a plain Python function, a Pydantic schema-backed callable, or an integration exposed through MCP.

Use this section when you need domain-specific tool patterns. For the core agent tool interface, start with the [tool system guide](../swarms/tools/main.md), [BaseTool reference](../swarms/tools/base_tool.md), and [MCP client utilities](../swarms/tools/mcp_client_call.md).

## When to Use a Tool

Add a tool when an agent needs a capability that should not be guessed by the language model:

- Fetching current or private data from an API
- Running a calculation with deterministic output
- Looking up documents, market data, search results, or social posts
- Sending data to another service with explicit inputs
- Wrapping an internal workflow behind a stable function call

Keep tools narrow. A tool should do one job, accept typed inputs, and return a predictable string or serialized result that the agent can reason over.

## Basic Pattern

```python
import os
import requests


def fetch_public_data(query: str) -> str:
    """Fetch public data for an agent workflow."""
    api_key = os.getenv("PUBLIC_DATA_API_KEY")
    response = requests.get(
        "https://api.example.com/search",
        params={"q": query, "api_key": api_key},
        timeout=20,
    )
    response.raise_for_status()
    return response.text
```

Then pass the callable to an agent:

```python
from swarms import Agent

agent = Agent(
    agent_name="Research-Agent",
    model_name="gpt-4.1",
    tools=[fetch_public_data],
)

agent.run("Find recent public data about battery supply chains.")
```

## Implementation Checklist

- Use clear function names that describe the external action.
- Add type hints for every parameter.
- Read secrets from environment variables with `os.getenv`.
- Set request timeouts for network calls.
- Return strings, JSON strings, or concise serialized results.
- Include enough error handling for the agent to understand failures.
- Keep side effects explicit in the function name and docstring.

For contribution standards, see [Contributing Tools and Plugins](../contributors/tools.md).
