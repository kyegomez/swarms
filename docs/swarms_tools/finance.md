# Finance Tools

Finance tools let Swarms agents retrieve market data, normalize financial records, and run deterministic calculations before producing a recommendation. They are useful for research assistants, analyst copilots, reporting agents, and monitoring workflows where the model needs fresh numbers instead of relying on stale training data.

Financial tools should be designed conservatively. Separate data retrieval from interpretation, return timestamps and sources, and make calculations explicit. The agent can explain tradeoffs, but a tool should provide the auditable inputs.

## Common Use Cases

- Retrieve quote, volume, market-cap, or fundamentals data from a provider.
- Summarize a watchlist with current prices and percentage moves.
- Calculate allocation drift, exposure, drawdown, or risk budget.
- Build a report from transaction exports or account snapshots.
- Compare market news against a portfolio or sector list.

Avoid tools that execute trades unless you have a tested confirmation flow, account-level permissions, and operational monitoring. A research tool and a trading tool should never share the same callable.

## Read-Only Market Snapshot

The example below shows a read-only finance tool with a stable output shape. Replace the mocked data access with your provider client.

```python
from datetime import datetime, timezone
from swarms import Agent


def get_market_snapshot(symbols: list[str]) -> dict:
    """Return current market snapshot data for a list of symbols."""
    normalized = [symbol.upper().strip() for symbol in symbols]
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "provider": "example-market-data-provider",
        "symbols": [
            {
                "symbol": symbol,
                "price": 100.0,
                "currency": "USD",
                "day_change_percent": 0.8,
                "source": "mock",
            }
            for symbol in normalized
        ],
    }


agent = Agent(
    agent_name="Market Research Agent",
    system_prompt=(
        "Use market tools for current prices. "
        "Mention the provider and checked_at timestamp when summarizing results."
    ),
    tools=[get_market_snapshot],
    max_loops=2,
)

print(agent.run("Check AAPL, MSFT, and NVDA and summarize the biggest mover."))
```

This pattern keeps the tool read-only and gives the final answer enough metadata for the user to judge freshness.

## Deterministic Risk Calculation

When the result is math-heavy, put the calculation in the tool. The agent can describe the result, but the tool should compute it.

```python
def calculate_position_risk(position_value: float, portfolio_value: float, max_weight: float = 0.15) -> dict:
    """Calculate position weight and whether it exceeds a maximum allocation."""
    if portfolio_value <= 0:
        raise ValueError("portfolio_value must be greater than zero")

    weight = position_value / portfolio_value
    excess_value = max(0.0, position_value - (portfolio_value * max_weight))

    return {
        "position_value": position_value,
        "portfolio_value": portfolio_value,
        "weight": weight,
        "max_weight": max_weight,
        "is_over_limit": weight > max_weight,
        "excess_value": excess_value,
    }
```

Use this style for allocation, exposure, value-at-risk approximations, tax lots, or drawdown math. It prevents the agent from doing arithmetic in free text.

## Credential Handling

Most market data providers require keys. Load them from environment variables and report missing configuration clearly.

```python
import os


def get_finance_api_key() -> str:
    api_key = os.getenv("FINANCIAL_DATA_API_KEY")
    if not api_key:
        raise RuntimeError("Set FINANCIAL_DATA_API_KEY before running finance tools")
    return api_key
```

Recommended environment variables:

- `FINANCIAL_DATA_API_KEY` for a generic market data provider.
- `NEWS_API_KEY` for financial news enrichment.
- `PORTFOLIO_EXPORT_PATH` for local CSV-based portfolio analysis.
- `FINANCE_TOOL_TIMEOUT_SECONDS` for network-bound tools.

Do not return secrets from a tool. If the agent needs to confirm configuration, return a boolean such as `{"configured": true, "provider": "..."}`.

## Portfolio Report Pattern

A portfolio report agent usually needs three tools:

| Tool | Responsibility |
| --- | --- |
| `load_positions` | Read positions from a trusted export or API. |
| `get_market_snapshot` | Retrieve current prices and market movement. |
| `calculate_exposure` | Compute allocations, concentration, and drift. |

The agent prompt should require source attribution:

```python
system_prompt = """
You are a portfolio reporting agent.
Use tools for all numeric values.
Do not invent prices, balances, or returns.
In the final answer, include the checked_at timestamp and any rows that could not be priced.
"""
```

## Safety Guidelines

- Treat all prices as time-sensitive. Include `checked_at`.
- Distinguish realized account data from derived calculations.
- Return provider errors instead of substituting model guesses.
- Keep trade execution out of research agents.
- Add tests for edge cases such as empty portfolios, zero balances, missing symbols, and unsupported currencies.

## Example Output Contract

Finance tools work best when every result follows a predictable contract:

```json
{
  "ok": true,
  "checked_at": "2026-01-01T00:00:00Z",
  "provider": "example-market-data-provider",
  "data": [],
  "warnings": []
}
```

If `ok` is false, the agent should stop and explain what failed instead of producing a market conclusion from incomplete data.

