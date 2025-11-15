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

    Example:
        ```python
        # Get all services
        result = await query_x402_services()
        print(f"Found {len(result['items'])} services")

        # Get first 10 services under 100000 atomic units
        result = await query_x402_services(limit=10, max_price=100000)
        ```
    """
    url = f"{base_url}/platform/v2/x402/discovery/resources"
    params = {"offset": offset}

    # If both limit and max_price are specified, fetch more services to account for filtering
    # This ensures we can return the requested number after filtering by price
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


def filter_services_by_price(
    services: List[Dict[str, Any]], max_price: int
) -> List[Dict[str, Any]]:
    """
    Filter services by maximum price in atomic units.

    Args:
        services: List of service dictionaries from the API.
        max_price: Maximum price in atomic units. Only services with at least one
                   payment option where maxAmountRequired <= max_price will be included.

    Returns:
        List of filtered service dictionaries.

    Example:
        ```python
        all_services = result["items"]
        affordable = filter_services_by_price(all_services, max_price=100000)
        ```
    """
    filtered = []
    for item in services:
        accepts = item.get("accepts", [])
        for accept in accepts:
            max_amount_str = accept.get("maxAmountRequired", "")
            if max_amount_str:
                try:
                    max_amount = int(max_amount_str)
                    if max_amount <= max_price:
                        filtered.append(item)
                        break  # Only add item once if any payment option matches
                except (ValueError, TypeError):
                    continue
    return filtered


def limit_services(
    services: List[Dict[str, Any]], max_count: int
) -> List[Dict[str, Any]]:
    """
    Limit the number of services returned.

    Args:
        services: List of service dictionaries.
        max_count: Maximum number of services to return.

    Returns:
        List containing at most max_count services.

    Example:
        ```python
        all_services = result["items"]
        limited = limit_services(all_services, max_count=10)
        ```
    """
    return services[:max_count]


async def get_x402_services(
    limit: Optional[int] = None,
    max_price: Optional[int] = None,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Get x402 services with optional filtering by count and price.

    This is a convenience function that queries the API and applies filters.

    Args:
        limit: Optional maximum number of services to return.
        max_price: Optional maximum price in atomic units to filter by.
        offset: Pagination offset for the API request. Defaults to 0.

    Returns:
        List of service dictionaries matching the criteria.

    Example:
        ```python
        # Get first 10 services under $0.10 USDC (100000 atomic units with 6 decimals)
        services = await get_x402_services(limit=10, max_price=100000)
        for service in services:
            print(service["resource"])
        ```
    """
    result = await query_x402_services(
        limit=limit, max_price=max_price, offset=offset
    )

    return result.get("items", [])


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

    Example:
        ```python
        # Get first 10 services under $0.10 USDC
        services_str = get_x402_services_sync(limit=10, max_price=100000)
        print(services_str)
        ```
    """
    services = asyncio.run(
        get_x402_services(
            limit=limit, max_price=max_price, offset=offset
        )
    )
    return str(services)



agent = Agent(
    agent_name="X402-Discovery-Agent",
    agent_description="A agent that queries the x402 discovery services from the Coinbase CDP API.",
    model_name="claude-haiku-4-5",
    dynamic_temperature_enabled=True,
    max_loops=1,
    dynamic_context_window=True,
    tools=[get_x402_services_sync],
    top_p=None,
    # temperature=0.0,
    temperature=None,
    tool_call_summary=True,
)

if __name__ == "__main__":

    # Run the agent
    out = agent.run(
        task="Summarize the first 10 services under 100000 atomic units (e.g., $0.10 USDC)"
    )
    print(out)