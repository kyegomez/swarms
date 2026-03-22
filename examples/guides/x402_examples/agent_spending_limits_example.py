"""
Example: Per-agent spending limits for x402 marketplace.

Demonstrates how to use PaymentPolicyEngine to:
- Enforce per-agent daily budgets
- Block or require approval for large payments
- Use circuit breakers to pause payments to failing facilitators
- Maintain an immutable audit trail
"""

import os

from dotenv import load_dotenv

from swarms import Agent
from swarms.utils.x402_spending_limits import (
    AgentBudget,
    PaymentPolicyEngine,
    SpendingRule,
)

load_dotenv()


# ---------------------------------------------------------------------------
# 1. Build the payment policy
# ---------------------------------------------------------------------------

engine = PaymentPolicyEngine()

# Per-agent daily spending limits
engine.add_budget(AgentBudget(agent_id="research-bot", max_amount=500.0, currency="USDC", window="daily"))
engine.add_budget(AgentBudget(agent_id="writer-bot", max_amount=100.0, currency="USDC", window="daily"))

# Global rules evaluated in order:
#   1. Block any single payment above 1000 USDC
#   2. Require human approval for payments above 100 USDC
#   3. Allow everything else
engine.add_rule(SpendingRule(rule_type="block_above", amount=1000.0, currency="USDC"))
engine.add_rule(SpendingRule(rule_type="require_approval_above", amount=100.0, currency="USDC"))
engine.add_rule(SpendingRule(rule_type="allow_all"))

# You can also load the same policy from a dict (mirrors PaySentry schema):
# engine.load_policy({
#     "budgets": [
#         {"agent_id": "research-bot", "window": "daily", "max_amount": 500, "currency": "USDC"},
#         {"agent_id": "writer-bot",   "window": "daily", "max_amount": 100, "currency": "USDC"},
#     ],
#     "rules": [
#         {"type": "block_above",            "amount": 1000, "currency": "USDC"},
#         {"type": "require_approval_above", "amount": 100,  "currency": "USDC"},
#         {"type": "allow_all"},
#     ],
# })


# ---------------------------------------------------------------------------
# 2. Wrapper that gates x402 payments through the policy engine
# ---------------------------------------------------------------------------

FACILITATOR_URL = "https://x402.org/facilitator"


def execute_x402_payment_with_policy(
    agent_id: str,
    amount: float,
    currency: str = "USDC",
    facilitator_url: str = FACILITATOR_URL,
) -> bool:
    """
    Check the policy engine before executing an x402 payment.

    In a real integration this function would call the x402 client SDK
    after the policy check passes.  Returns True if payment succeeded.
    """
    decision = engine.check_payment(
        agent_id=agent_id,
        amount=amount,
        currency=currency,
        facilitator_url=facilitator_url,
    )

    if not decision.approved:
        print(f"[BLOCKED] {agent_id}: {decision.reason}")
        return False

    if decision.requires_approval:
        # In production: send to approval queue / human review system
        print(f"[PENDING APPROVAL] {agent_id}: {decision.reason}")
        return False

    print(f"[ALLOWED] {agent_id}: paying {amount} {currency}")

    # --- Execute actual x402 payment here ---
    # Example (requires x402 + eth_account installed):
    #
    # from eth_account import Account
    # from x402.clients.httpx import x402HttpxClient
    # import asyncio, httpx
    #
    # async def _pay():
    #     account = Account.from_key(os.getenv("X402_PRIVATE_KEY"))
    #     async with x402HttpxClient(account=account, base_url=facilitator_url) as client:
    #         resp = await client.post("/settle", json={"amount": amount, "currency": currency})
    #         return resp.status_code == 200
    #
    # success = asyncio.run(_pay())
    success = True  # placeholder

    # Record outcome (updates spend tracker + circuit breaker + audit log)
    engine.record_payment(
        agent_id=agent_id,
        amount=amount,
        currency=currency,
        facilitator_url=facilitator_url,
        tx_hash="0xsimulated",
        success=success,
    )

    if budget_remaining := decision.budget_remaining:
        print(f"  Budget remaining after payment: {budget_remaining - amount:.4f} {currency}")

    return success


# ---------------------------------------------------------------------------
# 3. Create agents that use the policy-gated payment function
# ---------------------------------------------------------------------------

researcher = Agent(
    agent_name="research-bot",
    system_prompt="Research topics and compile summaries.",
    model_name="gpt-4o",
    max_loops=1,
)

writer = Agent(
    agent_name="writer-bot",
    system_prompt="Write polished articles based on research.",
    model_name="gpt-4o",
    max_loops=1,
)


# ---------------------------------------------------------------------------
# 4. Demo: simulate several payments to show policy enforcement
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Per-agent x402 Spending Limits Demo ===\n")

    # Normal payments — should be allowed
    execute_x402_payment_with_policy("research-bot", 25.0)
    execute_x402_payment_with_policy("writer-bot", 50.0)

    # Payment requiring approval (> 100 USDC rule)
    execute_x402_payment_with_policy("research-bot", 150.0)

    # Payment that exceeds block_above threshold
    execute_x402_payment_with_policy("research-bot", 1500.0)

    # Simulate budget exhaustion for writer-bot
    # writer-bot has only 100 USDC/day and already spent 50
    execute_x402_payment_with_policy("writer-bot", 60.0)  # 50+60=110 > 100 → blocked

    # Simulate circuit breaker tripping
    print("\n--- Simulating facilitator failures ---")
    for _ in range(5):
        engine.circuit_breaker.record_failure(FACILITATOR_URL)
    execute_x402_payment_with_policy("research-bot", 10.0)  # blocked by open circuit

    # Print audit log
    print("\n--- Audit Log ---")
    for entry in engine.audit_log.as_dicts():
        print(
            f"  {entry['timestamp']}  {entry['agent_id']:15s}  "
            f"{entry['amount']:8.2f} {entry['currency']}  "
            f"{'OK' if entry['success'] else 'FAIL'}"
        )

    # Print spend summary
    print("\n--- Spend Summary ---")
    for agent_id in ("research-bot", "writer-bot"):
        summary = engine.agent_spend_summary(agent_id)
        for window, info in summary["windows"].items():
            print(
                f"  {agent_id}: {info['spent']:.4f} / {info['limit']} "
                f"{info['currency']} ({window})"
            )
