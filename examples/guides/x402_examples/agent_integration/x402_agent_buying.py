from eth_account import Account
from swarms.payments import (
    AuditLogger,
    Budget,
    CircuitBreaker,
    PolicyEngine,
    PolicyRule,
    SpendTracker,
    X402PolicyClient,
    X402PolicyConfig,
)


import os
from dotenv import load_dotenv

load_dotenv()


async def buy_x402_service(
    base_url: str = None,
    endpoint: str = None,
    amount_atomic: int = 90000,
    agent_id: str = "x402-buyer",
    facilitator_url: str = "https://x402.org/facilitator",
):
    """
    Purchase a service from the X402 bazaar using the provided affordable_service details.

    This function sets up an X402 client with the user's private key, connects to the service provider,
    and executes a GET request to the service's endpoint as part of the buying process.

    Args:
        base_url (str, optional): The base URL of the service provider. Defaults to None.
        endpoint (str, optional): The specific API endpoint to interact with. Defaults to None.
        amount_atomic (int, optional): Amount in atomic units (e.g., 90000 = $0.09 USDC).
        agent_id (str, optional): Agent identifier for per-agent spending limits.
        facilitator_url (str, optional): Facilitator endpoint for circuit breaker tracking.

    Returns:
        response (httpx.Response): The response object returned by the GET request to the service endpoint.

    Example:
        ```python
        affordable_service = {"id": "service123", "price": 90000}
        response = await buy_x402_service(
            affordable_service,
            base_url="https://api.cdp.coinbase.com",
            endpoint="/x402/v1/bazaar/services/service123"
        )
        print(await response.aread())
        ```
    """
    key = os.getenv("X402_PRIVATE_KEY")

    # Set up your payment account from private key
    account = Account.from_key(key)

    spend_tracker = SpendTracker()
    policy_engine = PolicyEngine(
        rules=[
            PolicyRule(type="block_above", amount=1_000_000, currency="USDC"),
            PolicyRule(
                type="require_approval_above",
                amount=100_000,
                currency="USDC",
            ),
            PolicyRule(type="allow_all"),
        ],
        budgets=[
            Budget(
                agent_id=agent_id,
                window="daily",
                max_amount=500_000,
                currency="USDC",
            )
        ],
        spend_tracker=spend_tracker,
    )
    circuit_breaker = CircuitBreaker(
        failure_threshold=5,
        recovery_timeout_seconds=30,
    )
    audit_logger = AuditLogger()
    policy_config = X402PolicyConfig(
        agent_id=agent_id,
        currency="USDC",
        default_window="daily",
    )

    async with X402PolicyClient(
        account=account,
        base_url=base_url,
        policy_engine=policy_engine,
        spend_tracker=spend_tracker,
        circuit_breaker=circuit_breaker,
        audit_logger=audit_logger,
        policy_config=policy_config,
    ) as client:
        response = await client.get(
            endpoint,
            amount=amount_atomic,
            facilitator=facilitator_url,
        )
        print(await response.aread())

    return response
