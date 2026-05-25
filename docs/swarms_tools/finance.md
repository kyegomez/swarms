# Finance Tools

Finance tools let Swarms agents retrieve market data, portfolio context, filings, risk signals, or other financial inputs before producing an analysis. They are useful when an agent needs live or structured data instead of relying on model memory.

For a working example that uses `swarms-tools`, see the [Yahoo Finance example](../swarms/examples/yahoo_finance.md).

## Common Use Cases

- Looking up public market data for a ticker
- Summarizing portfolio or watchlist metrics
- Fetching macroeconomic indicators from a provider API
- Comparing assets against a fixed scoring rubric
- Pulling price, volume, or fundamentals data before an agent writes a report

## Example

```python
import os
import requests


def get_market_snapshot(symbol: str) -> str:
    """Return a compact market snapshot for a ticker symbol."""
    api_key = os.getenv("MARKET_DATA_API_KEY")
    response = requests.get(
        "https://api.example.com/markets/snapshot",
        params={"symbol": symbol, "api_key": api_key},
        timeout=20,
    )
    response.raise_for_status()
    return response.text
```

```python
from swarms import Agent

finance_agent = Agent(
    agent_name="Finance-Agent",
    model_name="gpt-4.1",
    tools=[get_market_snapshot],
)

finance_agent.run("Analyze the current risk profile for NVDA.")
```

## Notes

- Document the required provider and environment variables.
- Return the raw provider payload only when the downstream agent needs all fields.
- Prefer compact JSON for repeatable analysis workflows.
- Include provider timestamps when the freshness of the data matters.
- Avoid hardcoding API keys, account identifiers, or private portfolio values.
