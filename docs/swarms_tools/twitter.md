# Twitter Tools

Twitter tools help Swarms agents retrieve posts, profile context, engagement signals, or social conversation data from an approved provider API. They are useful for brand monitoring, market research, creator analytics, and incident response workflows.

Use the patterns in this page with the general [Swarms tool system](../swarms/tools/main.md). Always follow the API provider's terms and privacy requirements.

## Common Use Cases

- Monitoring recent posts for a keyword or account
- Summarizing public conversation around a launch
- Tracking engagement changes for a campaign
- Routing social mentions to a support or research agent
- Comparing public sentiment across topics

## Example

```python
import os
import requests


def search_recent_posts(query: str, limit: int = 10) -> str:
    """Return recent public posts matching a query."""
    bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
    response = requests.get(
        "https://api.example.com/twitter/search",
        params={"query": query, "limit": limit},
        headers={"Authorization": f"Bearer {bearer_token}"},
        timeout=20,
    )
    response.raise_for_status()
    return response.text
```

```python
from swarms import Agent

social_agent = Agent(
    agent_name="Social-Research-Agent",
    model_name="gpt-4.1",
    tools=[search_recent_posts],
)

social_agent.run("Summarize recent public posts about our product launch.")
```

## Safety and Compliance

- Use official or approved provider APIs.
- Do not store access tokens in code or documentation examples.
- Return only the fields needed for the agent's task.
- Respect rate limits and provider policies.
- Avoid collecting private, sensitive, or unnecessary personal data.
