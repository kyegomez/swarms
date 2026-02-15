# Payments Policy Engine

The payments module provides policy-driven spend control primitives for agentic workflows. It is designed to let you evaluate spend requests before execution and enforce budget/rule constraints consistently.

## Core Types

- `PolicyRule`: Rule object for behavior like `block_above`, `require_approval_above`, and `allow_all`.
- `Budget`: Windowed budget guardrail per agent and currency.
- `PolicyDecision`: Evaluation result with `status`, `reason`, optional `rule_id`, and `metadata`.
- `SpendTracker`: Tracks spend by window (`hourly`, `daily`, `monthly`).
- `PolicyEngine`: Applies budget checks first, then rule checks.

## Evaluation Flow

`PolicyEngine.evaluate(agent_id, amount, currency)` evaluates in this order:

1. **Budget checks** for matching `agent_id` and `currency`
2. **Rules checks** in declared order
3. **Default allow** when no blocking or approval rule applies

Possible statuses:

- `allowed`
- `requires_approval`
- `blocked`

## Quickstart

```python
from swarms.payments.policy import Budget, PolicyEngine, PolicyRule

rules = [
    PolicyRule(type="require_approval_above", amount=100.0, currency="USD", rule_id="approval-100"),
    PolicyRule(type="block_above", amount=500.0, currency="USD", rule_id="block-500"),
]

budgets = [
    Budget(agent_id="research-agent", window="daily", max_amount=1000.0, currency="USD")
]

engine = PolicyEngine(rules=rules, budgets=budgets)

decision = engine.evaluate(
    agent_id="research-agent",
    amount=120.0,
    currency="USD",
)

print(decision.status, decision.reason, decision.metadata)
```

## Recording Spend

After allowed execution, record realized spend:

```python
engine.record_spend(
    agent_id="research-agent",
    amount=42.5,
    currency="USD",
    window="daily",
)
```

## Notes

- Keep rules explicit and deterministic.
- Use one currency normalization strategy before evaluation.
- Place stricter rules before permissive rules such as `allow_all`.
