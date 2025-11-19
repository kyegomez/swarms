# X402 Discovery Query Agent

This example demonstrates how to create a Swarms agent that can search and query services from the X402 bazaar using the Coinbase CDP API. The agent can discover available services, filter them by price, and provide summaries of the results.

## Overview

The X402 Discovery Query Agent enables you to:

| Feature | Description |
|---------|-------------|
| Query X402 services | Search the X402 bazaar for available services |
| Filter by price | Find services within your budget |
| Summarize results | Get AI-powered summaries of discovered services |
| Pagination support | Handle large result sets efficiently |

## Prerequisites

Before you begin, ensure you have:

- Python 3.10 or higher
- API keys for your AI model provider (e.g., Anthropic Claude)
- `httpx` library for async HTTP requests

## Installation

Install the required dependencies:

```bash
pip install swarms httpx
```

## Code Example

Here's the complete implementation of the X402 Discovery Query Agent:

```python
import asyncio
from typing import List, Optional, Dict, Any
from swarms import Agent
import httpx


async def query_x402_services(
    limit: Optional[int] = None,
    max_price: Optional[int] = None,
    offset: int = 0,
    base_url: str = "https://api.cdp.coinbase.com",
) -> Dict[str, Any]:
    """
    Query x402 discovery services from the Coinbase CDP API.

    Args:
        limit: Optional maximum number of services to return. If None, returns all available.
        max_price: Optional maximum price in atomic units to filter by. Only services with
                   maxAmountRequired <= max_price will be included.
        offset: Pagination offset for the API request. Defaults to 0.
        base_url: Base URL for the API. Defaults to Coinbase CDP API.

    Returns:
        Dict containing the API response with 'items' list and pagination info.

    Raises:
        httpx.HTTPError: If the HTTP request fails.
        httpx.RequestError: If there's a network error.
    """
    url = f"{base_url}/platform/v2/x402/discovery/resources"
    params = {"offset": offset}

    # If both limit and max_price are specified, fetch more services to account for filtering
    api_limit = limit
    if limit is not None and max_price is not None:
        # Fetch 5x the limit to account for services that might be filtered out
        api_limit = limit * 5

    if api_limit is not None:
        params["limit"] = api_limit

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

    # Filter by price if max_price is specified
    if max_price is not None and "items" in data:
        filtered_items = []
        for item in data.get("items", []):
            # Check if any payment option in 'accepts' has maxAmountRequired <= max_price
            accepts = item.get("accepts", [])
            for accept in accepts:
                max_amount_str = accept.get("maxAmountRequired", "")
                if max_amount_str:
                    try:
                        max_amount = int(max_amount_str)
                        if max_amount <= max_price:
                            filtered_items.append(item)
                            break  # Only add item once if any payment option matches
                    except (ValueError, TypeError):
                        continue

        # Apply limit to filtered results if specified
        if limit is not None:
            filtered_items = filtered_items[:limit]

        data["items"] = filtered_items
        # Update pagination total if we filtered
        if "pagination" in data:
            data["pagination"]["total"] = len(filtered_items)

    return data


def get_x402_services_sync(
    limit: Optional[int] = None,
    max_price: Optional[int] = None,
    offset: int = 0,
) -> str:
    """
    Synchronous wrapper for get_x402_services that returns a formatted string.

    Args:
        limit: Optional maximum number of services to return.
        max_price: Optional maximum price in atomic units to filter by.
        offset: Pagination offset for the API request. Defaults to 0.

    Returns:
        JSON-formatted string of service dictionaries matching the criteria.
    """
    async def get_x402_services():
        result = await query_x402_services(
            limit=limit, max_price=max_price, offset=offset
        )
        return result.get("items", [])
    
    services = asyncio.run(get_x402_services())
    return str(services)


# Initialize the agent with the discovery tool
agent = Agent(
    agent_name="X402-Discovery-Agent",
    agent_description="A agent that queries the x402 discovery services from the Coinbase CDP API.",
    model_name="claude-haiku-4-5",
    dynamic_temperature_enabled=True,
    max_loops=1,
    dynamic_context_window=True,
    tools=[get_x402_services_sync],
    top_p=None,
    temperature=None,
    tool_call_summary=True,
)

if __name__ == "__main__":
    # Run the agent
    out = agent.run(
        task="Summarize the first 10 services under 100000 atomic units (e.g., $0.10 USDC)"
    )
    print(out)
```

## Usage

### Basic Query

Query all available services:

```python
result = await query_x402_services()
print(f"Found {len(result['items'])} services")
```

### Filtered Query

Get services within a specific price range:

```python
# Get first 10 services under 100000 atomic units ($0.10 USDC with 6 decimals)
services = await get_x402_services(limit=10, max_price=100000)
for service in services:
    print(service["resource"])
```

### Using the Agent

Run the agent to get AI-powered summaries:

```python
# The agent will automatically call the tool and provide a summary
out = agent.run(
    task="Find and summarize 5 affordable services under 50000 atomic units"
)
print(out)
```

## Understanding Price Units

X402 services use atomic units for pricing. For example:

- **USDC** typically uses 6 decimals
- 100,000 atomic units = $0.10 USDC
- 1,000,000 atomic units = $1.00 USDC

Always check the `accepts` array in each service to understand the payment options and their price requirements.

## API Response Structure

Each service in the response contains:

- `resource`: The service endpoint or resource identifier
- `accepts`: Array of payment options with `maxAmountRequired` values
- Additional metadata about the service

## Error Handling

The functions handle various error cases:

- Network errors are raised as `httpx.RequestError`
- HTTP errors are raised as `httpx.HTTPError`
- Invalid price values are silently skipped during filtering

## Next Steps

1. Customize the agent's system prompt for specific use cases
2. Add additional filtering criteria (e.g., by service type)
3. Implement caching for frequently accessed services
4. Create a web interface for browsing services
5. Integrate with payment processing to actually use discovered services

## Related Documentation

- [X402 Payment Integration](x402_payment_integration.md) - Learn how to monetize your agents with X402
- [Agent Tools Reference](../swarms/tools/tools_examples.md) - Understand how to create and use tools with agents
