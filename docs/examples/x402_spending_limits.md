# X402 Spending Limits and Circuit Breakers

This guide shows how to add per-agent spending limits, approval thresholds, circuit breakers, and audit logs around x402 payments.

## Why use spending limits?

- Prevent a single agent from draining shared funds
- Enforce daily or hourly budgets per agent
- Pause payments to failing facilitators
- Keep an audit trail of payment decisions

## Example: Guarded X402 purchase

```python
import os
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

key = os.getenv("X402_PRIVATE_KEY")
account = Account.from_key(key)

spend_tracker = SpendTracker()
policy_engine = PolicyEngine(
    rules=[
        PolicyRule(type="block_above", amount=1_000_000, currency="USDC"),
        PolicyRule(type="require_approval_above", amount=100_000, currency="USDC"),
        PolicyRule(type="allow_all"),
    ],
    budgets=[
        Budget(agent_id="research-bot", window="daily", max_amount=500_000, currency="USDC")
    ],
    spend_tracker=spend_tracker,
)

circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout_seconds=30)
audit_logger = AuditLogger()
policy_config = X402PolicyConfig(agent_id="research-bot", currency="USDC", default_window="daily")

async def purchase():
    async with X402PolicyClient(
        account=account,
        base_url="https://api.cdp.coinbase.com",
        policy_engine=policy_engine,
        spend_tracker=spend_tracker,
        circuit_breaker=circuit_breaker,
        audit_logger=audit_logger,
        policy_config=policy_config,
    ) as client:
        response = await client.get(
            "/x402/v1/bazaar/services/service123",
            amount=90_000,
            facilitator="https://x402.org/facilitator",
        )
        return await response.aread()
```

## Notes

- Amounts use atomic units (USDC has 6 decimals).
- Policies are evaluated before the payment request.
- Circuit breakers open after repeated 5xx responses.
- Audit logs are collected in memory; you can persist them in your own storage.
