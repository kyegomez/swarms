# X402 Discovery Query

This guide shows the basic pattern for querying x402 discovery services from a Swarms agent tool. Use it when an agent needs to list available paid services before deciding whether to purchase or call one.

For the broader agent walkthrough, see [X402 Tools Agent](x402_tools_agent.md).

## Install Dependencies

```bash
pip install swarms httpx python-dotenv
```

## Query Available Services

```python
import asyncio
import httpx


async def query_x402_services(limit: int = 10, offset: int = 0):
    url = "https://api.cdp.coinbase.com/platform/v2/x402/discovery/resources"
    params = {"limit": limit, "offset": offset}

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


services = asyncio.run(query_x402_services(limit=5))
print(services)
```

## Add a Tool Wrapper

Wrap the query in a synchronous function before passing it to an agent.

```python
import asyncio


def get_x402_services_sync(limit: int = 10, offset: int = 0):
    return asyncio.run(query_x402_services(limit=limit, offset=offset))
```

Then attach the wrapper to an agent through the same tool patterns used in [Agent with Tools](agent_with_tools.md).

## Practical Tips

- Keep discovery queries read-only.
- Limit results during development so responses stay small.
- Validate service metadata before attempting a purchase flow.
- Log the selected service ID, endpoint, and price before any paid call.
- Keep wallet or private-key handling separate from discovery.

## Related Guides

- [X402 Tools Agent](x402_tools_agent.md)
- [Agent with Tools](agent_with_tools.md)
- [Tools Reference](../tools/main.md)
- [Multi MCP Agent](multi_mcp_agent.md)
