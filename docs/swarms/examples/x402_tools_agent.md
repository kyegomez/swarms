# X402 Tools Agent Tutorial

This page provides practical examples of how to create agents with X402 tools for discovering services in the bazaar, purchasing services, managing wallets, and more.

## Overview

X402 is a payment protocol that enables cryptocurrency payments for API endpoints. With X402 tools, your agents can:

- **Discover services** in the X402 bazaar
- **Purchase services** using cryptocurrency
- **Manage wallet** operations
- **Query service details** and pricing
- **Filter services** by price and features

## Prerequisites

- Python 3.10+
- An EVM-compatible wallet (MetaMask, Coinbase Wallet, etc.)
- Coinbase CDP API credentials (for mainnet)
- X402 library installed

## Installation

```bash
pip install swarms x402 eth-account httpx python-dotenv
```

## Environment Setup

Create a `.env` file:

```bash
# Your wallet private key (keep this secure!)
X402_PRIVATE_KEY=your_private_key_here

# Coinbase CDP API credentials (for mainnet)
CDP_API_KEY_NAME=your_api_key_name
CDP_API_KEY_SECRET=your_api_key_secret

# OpenAI API key (for agent)
OPENAI_API_KEY=your_openai_key
```

## Example 1: X402 Discovery Agent

This agent can discover and query services from the X402 bazaar.

```python
import asyncio
from typing import List, Optional, Dict, Any
from swarms import Agent
import httpx
import json


async def query_x402_services(
    limit: Optional[int] = None,
    max_price: Optional[int] = None,
    offset: int = 0,
    base_url: str = "https://api.cdp.coinbase.com",
) -> Dict[str, Any]:
    """
    Query x402 discovery services from the Coinbase CDP API.

    Args:
        limit: Optional maximum number of services to return
        max_price: Optional maximum price in atomic units (e.g., 100000 = $0.10 USDC with 6 decimals)
        offset: Pagination offset. Defaults to 0
        base_url: Base URL for the API. Defaults to Coinbase CDP API

    Returns:
        Dict containing the API response with 'items' list and pagination info

    Example:
        >>> result = await query_x402_services(limit=10, max_price=100000)
        >>> print(f"Found {len(result['items'])} services")
    """
    url = f"{base_url}/platform/v2/x402/discovery/resources"
    params = {"offset": offset}

    if limit is not None:
        params["limit"] = limit * 5 if max_price else limit

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

    # Filter by price if specified
    if max_price is not None and "items" in data:
        filtered_items = []
        for item in data.get("items", []):
            accepts = item.get("accepts", [])
            for accept in accepts:
                max_amount_str = accept.get("maxAmountRequired", "")
                if max_amount_str:
                    try:
                        max_amount = int(max_amount_str)
                        if max_amount <= max_price:
                            filtered_items.append(item)
                            break
                    except (ValueError, TypeError):
                        continue

        if limit is not None:
            filtered_items = filtered_items[:limit]
        data["items"] = filtered_items

    return data


def get_x402_services_sync(
    limit: Optional[int] = None,
    max_price: Optional[int] = None,
    offset: int = 0,
) -> str:
    """
    Synchronous wrapper for querying x402 services.

    Args:
        limit: Optional maximum number of services to return
        max_price: Optional maximum price in atomic units
        offset: Pagination offset. Defaults to 0

    Returns:
        JSON-formatted string of service dictionaries

    Example:
        >>> services = get_x402_services_sync(limit=10, max_price=100000)
        >>> print(services)
    """
    services = asyncio.run(
        query_x402_services(limit=limit, max_price=max_price, offset=offset)
    )
    return json.dumps(services, indent=2)


def get_service_details(service_id: str) -> str:
    """
    Get detailed information about a specific x402 service.

    Args:
        service_id: The unique identifier of the service

    Returns:
        JSON-formatted string with service details

    Example:
        >>> details = get_service_details("service123")
    """
    # In production, this would query the CDP API for specific service details
    details = {
        "id": service_id,
        "name": f"Service {service_id}",
        "description": "Service description",
        "pricing": {"amount": "100000", "currency": "USDC"},
        "endpoint": f"https://api.example.com/services/{service_id}",
    }
    return json.dumps(details, indent=2)


# Initialize discovery agent
discovery_agent = Agent(
    agent_name="X402-Discovery-Agent",
    agent_description="Agent that queries and discovers x402 services from the bazaar",
    system_prompt="""You are an x402 service discovery agent. You can:
    - Query available services from the x402 bazaar
    - Filter services by price and features
    - Get detailed information about specific services
    - Help users find the best services for their needs
    
    Always provide clear summaries of services and their pricing.""",
    model_name="gpt-4o-mini",
    max_loops=3,
    tools=[
        get_x402_services_sync,
        get_service_details,
    ],
    verbose=True,
)

# Example usage
result = discovery_agent.run(
    "Find the first 5 services under $0.10 USDC and summarize them"
)
print(result)
```

## Example 2: X402 Service Purchase Agent

This agent can purchase services from the X402 bazaar using wallet integration.

```python
import os
from eth_account import Account
from x402.clients.httpx import x402HttpxClient
from swarms import Agent
from typing import Dict, Any
import json
from dotenv import load_dotenv

load_dotenv()


async def buy_x402_service(
    base_url: str,
    endpoint: str,
    private_key: str = None,
) -> str:
    """
    Purchase a service from the X402 bazaar.

    Args:
        base_url: The base URL of the service provider
        endpoint: The specific API endpoint to purchase
        private_key: Wallet private key (defaults to X402_PRIVATE_KEY env var)

    Returns:
        JSON-formatted string with purchase response

    Example:
        >>> result = await buy_x402_service(
        ...     base_url="https://api.example.com",
        ...     endpoint="/x402/v1/bazaar/services/service123"
        ... )
    """
    if private_key is None:
        private_key = os.getenv("X402_PRIVATE_KEY")
    
    if not private_key:
        return json.dumps({"error": "Private key not found"})

    # Set up payment account from private key
    account = Account.from_key(private_key)

    try:
        async with x402HttpxClient(account=account, base_url=base_url) as client:
            response = await client.get(endpoint)
            response_data = await response.aread()
            return json.dumps({
                "status": "success",
                "response": response_data.decode(),
                "wallet_address": account.address
            }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


def buy_service_sync(
    base_url: str,
    endpoint: str,
) -> str:
    """
    Synchronous wrapper for purchasing x402 services.

    Args:
        base_url: The base URL of the service provider
        endpoint: The specific API endpoint to purchase

    Returns:
        JSON-formatted string with purchase response
    """
    import asyncio
    return asyncio.run(buy_x402_service(base_url, endpoint))


def get_wallet_balance(wallet_address: str = None) -> str:
    """
    Get the balance of a wallet address.

    Args:
        wallet_address: Wallet address (defaults to wallet from private key)

    Returns:
        JSON-formatted string with wallet balance

    Example:
        >>> balance = get_wallet_balance("0x1234...")
    """
    if wallet_address is None:
        private_key = os.getenv("X402_PRIVATE_KEY")
        if private_key:
            account = Account.from_key(private_key)
            wallet_address = account.address
    
    # In production, this would query the blockchain for balance
    balance_info = {
        "wallet_address": wallet_address,
        "balance_usdc": "100.50",
        "balance_eth": "0.5",
        "network": "base-sepolia"
    }
    return json.dumps(balance_info, indent=2)


def check_service_affordability(
    service_price: int,
    wallet_address: str = None,
) -> str:
    """
    Check if the wallet has sufficient funds for a service.

    Args:
        service_price: Service price in atomic units
        wallet_address: Wallet address to check

    Returns:
        JSON-formatted string with affordability check result
    """
    balance_info = json.loads(get_wallet_balance(wallet_address))
    balance_usdc = float(balance_info.get("balance_usdc", 0))
    price_usdc = service_price / 1_000_000  # Assuming 6 decimals
    
    affordable = balance_usdc >= price_usdc
    
    result = {
        "affordable": affordable,
        "service_price_usdc": price_usdc,
        "wallet_balance_usdc": balance_usdc,
        "sufficient_funds": affordable
    }
    return json.dumps(result, indent=2)


# Initialize purchase agent
purchase_agent = Agent(
    agent_name="X402-Purchase-Agent",
    agent_description="Agent that can purchase services from the x402 bazaar using wallet integration",
    system_prompt="""You are an x402 service purchase agent with wallet access. You can:
    - Check wallet balance
    - Verify if services are affordable
    - Purchase services from the x402 bazaar
    - Manage wallet operations
    
    Always check wallet balance and affordability before making purchases.
    Only purchase services when the wallet has sufficient funds.""",
    model_name="gpt-4o-mini",
    max_loops=5,
    tools=[
        buy_service_sync,
        get_wallet_balance,
        check_service_affordability,
    ],
    verbose=True,
)

# Example usage
result = purchase_agent.run(
    "Check my wallet balance, then purchase the service at https://api.example.com/x402/v1/bazaar/services/service123"
)
print(result)
```

## Example 3: Complete X402 Agent with All Tools

This comprehensive agent combines discovery, purchase, wallet management, and service analysis.

```python
import os
import asyncio
import json
from typing import Optional, Dict, Any, List
from eth_account import Account
from x402.clients.httpx import x402HttpxClient
from swarms import Agent
import httpx
from dotenv import load_dotenv

load_dotenv()


# Tool 1: Discover services
async def discover_services(
    limit: Optional[int] = None,
    max_price: Optional[int] = None,
) -> Dict[str, Any]:
    """Discover x402 services from the bazaar."""
    url = "https://api.cdp.coinbase.com/platform/v2/x402/discovery/resources"
    params = {}
    if limit:
        params["limit"] = limit * 5 if max_price else limit
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    
    if max_price and "items" in data:
        filtered = [
            item for item in data.get("items", [])
            if any(
                int(accept.get("maxAmountRequired", "999999999"))
                <= max_price
                for accept in item.get("accepts", [])
            )
        ]
        data["items"] = filtered[:limit] if limit else filtered
    
    return data


def discover_services_sync(
    limit: Optional[int] = None,
    max_price: Optional[int] = None,
) -> str:
    """Synchronous wrapper for discovering services."""
    result = asyncio.run(discover_services(limit=limit, max_price=max_price))
    return json.dumps(result, indent=2)


# Tool 2: Get wallet info
def get_wallet_info() -> str:
    """Get wallet information from private key."""
    private_key = os.getenv("X402_PRIVATE_KEY")
    if not private_key:
        return json.dumps({"error": "Private key not found"})
    
    account = Account.from_key(private_key)
    info = {
        "wallet_address": account.address,
        "has_private_key": True,
        "network": "base-sepolia"
    }
    return json.dumps(info, indent=2)


# Tool 3: Check balance
def check_balance() -> str:
    """Check wallet balance."""
    private_key = os.getenv("X402_PRIVATE_KEY")
    if not private_key:
        return json.dumps({"error": "Private key not found"})
    
    account = Account.from_key(private_key)
    # In production, query blockchain
    balance = {
        "wallet_address": account.address,
        "usdc_balance": "100.00",
        "eth_balance": "0.5"
    }
    return json.dumps(balance, indent=2)


# Tool 4: Purchase service
async def purchase_service(
    base_url: str,
    endpoint: str,
) -> str:
    """Purchase a service from x402 bazaar."""
    private_key = os.getenv("X402_PRIVATE_KEY")
    if not private_key:
        return json.dumps({"error": "Private key not found"})
    
    account = Account.from_key(private_key)
    
    try:
        async with x402HttpxClient(account=account, base_url=base_url) as client:
            response = await client.get(endpoint)
            data = await response.aread()
            return json.dumps({
                "status": "success",
                "wallet": account.address,
                "response": data.decode()
            }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


def purchase_service_sync(base_url: str, endpoint: str) -> str:
    """Synchronous wrapper for purchasing services."""
    return asyncio.run(purchase_service(base_url, endpoint))


# Tool 5: Analyze service
def analyze_service(service_data: str) -> str:
    """Analyze a service and provide recommendations."""
    try:
        service = json.loads(service_data)
        analysis = {
            "service_id": service.get("id", "unknown"),
            "price_range": "Affordable" if service.get("price", 0) < 100000 else "Expensive",
            "recommendation": "Good value" if service.get("price", 0) < 100000 else "Consider alternatives"
        }
        return json.dumps(analysis, indent=2)
    except:
        return json.dumps({"error": "Invalid service data"}, indent=2)


# Tool 6: Filter affordable services
def filter_affordable_services(
    services_json: str,
    max_price: int = 100000,
) -> str:
    """Filter services by affordability."""
    try:
        data = json.loads(services_json)
        items = data.get("items", [])
        
        affordable = []
        for item in items:
            accepts = item.get("accepts", [])
            for accept in accepts:
                price = int(accept.get("maxAmountRequired", "999999999"))
                if price <= max_price:
                    affordable.append(item)
                    break
        
        return json.dumps({"affordable_services": affordable, "count": len(affordable)}, indent=2)
    except:
        return json.dumps({"error": "Invalid services data"}, indent=2)


# Tool 7: Get service recommendations
def recommend_services(services_json: str, budget: int = 100000) -> str:
    """Recommend services based on budget."""
    try:
        data = json.loads(services_json)
        items = data.get("items", [])
        
        recommendations = []
        for item in items[:10]:  # Top 10
            accepts = item.get("accepts", [])
            for accept in accepts:
                price = int(accept.get("maxAmountRequired", "999999999"))
                if price <= budget:
                    recommendations.append({
                        "id": item.get("id"),
                        "resource": item.get("resource"),
                        "price": price,
                        "reason": "Within budget"
                    })
                    break
        
        return json.dumps({"recommendations": recommendations}, indent=2)
    except:
        return json.dumps({"error": "Invalid services data"}, indent=2)


# Initialize comprehensive x402 agent
x402_agent = Agent(
    agent_name="X402-Comprehensive-Agent",
    agent_description="Comprehensive x402 agent with discovery, purchase, wallet, and analysis tools",
    system_prompt="""You are a comprehensive x402 agent with full access to:
    - Service discovery from the x402 bazaar
    - Wallet management and balance checking
    - Service purchasing capabilities
    - Service analysis and recommendations
    
    Workflow:
    1. Check wallet balance before any purchase
    2. Discover services that match user requirements
    3. Filter services by affordability
    4. Analyze and recommend best services
    5. Purchase services when user confirms
    
    Always verify wallet has sufficient funds before purchasing.
    Provide clear summaries of services and pricing.""",
    model_name="gpt-4o-mini",
    max_loops=10,
    tools=[
        discover_services_sync,
        get_wallet_info,
        check_balance,
        purchase_service_sync,
        analyze_service,
        filter_affordable_services,
        recommend_services,
    ],
    verbose=True,
)

# Example usage
result = x402_agent.run(
    "Check my wallet balance, discover services under $0.10 USDC, "
    "analyze them, and recommend the best 3 services for me"
)
print(result)
```

## Example 4: Wallet-Enabled Service Buyer

This agent focuses specifically on purchasing services with wallet integration.

```python
import os
import asyncio
import json
from eth_account import Account
from x402.clients.httpx import x402HttpxClient
from swarms import Agent
from dotenv import load_dotenv

load_dotenv()


def get_my_wallet_address() -> str:
    """Get the wallet address from the private key."""
    private_key = os.getenv("X402_PRIVATE_KEY")
    if not private_key:
        return json.dumps({"error": "Private key not found in environment"})
    
    account = Account.from_key(private_key)
    return json.dumps({
        "wallet_address": account.address,
        "status": "active"
    }, indent=2)


async def buy_service_with_wallet(
    service_url: str,
    service_endpoint: str,
) -> str:
    """
    Purchase a service using the agent's wallet.

    Args:
        service_url: Base URL of the service (e.g., "https://api.example.com")
        service_endpoint: Endpoint path (e.g., "/x402/v1/bazaar/services/123")

    Returns:
        JSON-formatted string with purchase result
    """
    private_key = os.getenv("X402_PRIVATE_KEY")
    if not private_key:
        return json.dumps({"error": "Wallet private key not configured"})
    
    account = Account.from_key(private_key)
    
    try:
        async with x402HttpxClient(account=account, base_url=service_url) as client:
            response = await client.get(service_endpoint)
            response_data = await response.aread()
            
            return json.dumps({
                "status": "purchased",
                "wallet": account.address,
                "service": service_endpoint,
                "response": response_data.decode()[:500]  # First 500 chars
            }, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "wallet": account.address
        }, indent=2)


def buy_service_sync_wrapper(service_url: str, service_endpoint: str) -> str:
    """Synchronous wrapper for buying services."""
    return asyncio.run(buy_service_with_wallet(service_url, service_endpoint))


def verify_purchase_affordability(service_price_atomic: int) -> str:
    """
    Verify if the wallet can afford a service.

    Args:
        service_price_atomic: Service price in atomic units

    Returns:
        JSON-formatted string with affordability check
    """
    private_key = os.getenv("X402_PRIVATE_KEY")
    if not private_key:
        return json.dumps({"error": "Wallet not configured"})
    
    account = Account.from_key(private_key)
    
    # In production, check actual balance from blockchain
    # For demo, assume we have sufficient funds
    price_usdc = service_price_atomic / 1_000_000  # 6 decimals
    
    result = {
        "wallet_address": account.address,
        "service_price_usdc": price_usdc,
        "can_afford": True,  # In production, check actual balance
        "recommendation": "Proceed with purchase" if True else "Insufficient funds"
    }
    return json.dumps(result, indent=2)


# Initialize wallet-enabled buyer agent
buyer_agent = Agent(
    agent_name="X402-Wallet-Buyer",
    agent_description="Agent with wallet access for purchasing x402 services",
    system_prompt="""You are a service purchasing agent with direct wallet access.
    
    Your capabilities:
    - Access your own wallet using the configured private key
    - Check wallet address and status
    - Verify affordability before purchases
    - Purchase services from x402 bazaar
    
    Always:
    1. Verify wallet is configured
    2. Check affordability before purchasing
    3. Confirm purchase details with user
    4. Execute purchase using wallet
    
    Never purchase without user confirmation or if wallet lacks funds.""",
    model_name="gpt-4o-mini",
    max_loops=5,
    tools=[
        get_my_wallet_address,
        buy_service_sync_wrapper,
        verify_purchase_affordability,
    ],
    verbose=True,
)

# Example usage
result = buyer_agent.run(
    "Show me my wallet address, then purchase the service at "
    "https://api.example.com/x402/v1/bazaar/services/research-agent-001"
)
print(result)
```

## Key Takeaways

1. **Wallet Integration**: Agents can access their own wallets using private keys stored in environment variables
2. **Service Discovery**: Use the CDP API to discover available x402 services in the bazaar
3. **Price Filtering**: Filter services by maximum price in atomic units (6 decimals for USDC)
4. **Purchase Flow**: Use `x402HttpxClient` with wallet account to purchase services
5. **Error Handling**: Always handle wallet and purchase errors gracefully
6. **Security**: Never hardcode private keys - always use environment variables
7. **Atomic Units**: X402 uses atomic units (e.g., 100000 = $0.10 USDC with 6 decimals)


For more detailed information about X402 integration, see:
- [X402 Payment Integration](../examples/x402_payment_integration.md)
- [X402 Discovery Query](../examples/x402_discovery_query.md)
- [X402 Documentation](https://docs.cdp.coinbase.com/x402)

