"""
Per-agent spending limits and payment policy enforcement for x402 marketplace.

Provides:
- AgentBudget: per-agent budget configuration with time windows
- SpendingRule: policy rules (block_above, require_approval_above, allow_all)
- SpendTracker: tracks cumulative spend per agent within time windows
- CircuitBreaker: auto-pauses payments to failing facilitators
- PaymentPolicyEngine: combines budget + rules + circuit breaker into one control plane
- AuditLog: immutable spending audit trail

Usage::

    from swarms.utils.x402_spending_limits import (
        PaymentPolicyEngine,
        AgentBudget,
        SpendingRule,
        SpendDecision,
    )

    engine = PaymentPolicyEngine()
    engine.add_budget(AgentBudget(
        agent_id="research-bot",
        max_amount=500.0,
        currency="USDC",
        window="daily",
    ))
    engine.add_rule(SpendingRule(rule_type="block_above", amount=1000.0, currency="USDC"))
    engine.add_rule(SpendingRule(rule_type="require_approval_above", amount=100.0, currency="USDC"))
    engine.add_rule(SpendingRule(rule_type="allow_all"))

    decision = engine.check_payment(agent_id="research-bot", amount=25.0, currency="USDC")
    if decision.approved:
        # proceed with payment
        engine.record_payment(
            agent_id="research-bot",
            amount=25.0,
            currency="USDC",
            facilitator_url="https://x402.org/facilitator",
            tx_hash="0xabc...",
            success=True,
        )
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional


class WindowType(str, Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


_WINDOW_SECONDS = {
    WindowType.HOURLY: 3600,
    WindowType.DAILY: 86400,
    WindowType.WEEKLY: 604800,
    WindowType.MONTHLY: 2592000,  # 30 days
}


class RuleType(str, Enum):
    BLOCK_ABOVE = "block_above"
    REQUIRE_APPROVAL_ABOVE = "require_approval_above"
    ALLOW_ALL = "allow_all"


@dataclass
class AgentBudget:
    """Per-agent spending budget within a rolling time window."""

    agent_id: str
    max_amount: float
    currency: str = "USDC"
    window: str = "daily"  # hourly | daily | weekly | monthly

    def window_seconds(self) -> int:
        key = WindowType(self.window)
        return _WINDOW_SECONDS[key]


@dataclass
class SpendingRule:
    """A single spending policy rule evaluated in order."""

    rule_type: str  # block_above | require_approval_above | allow_all
    amount: Optional[float] = None
    currency: Optional[str] = None

    def __post_init__(self) -> None:
        valid = {rt.value for rt in RuleType}
        if self.rule_type not in valid:
            raise ValueError(
                f"rule_type must be one of {valid}, got {self.rule_type!r}"
            )
        if self.rule_type != RuleType.ALLOW_ALL.value and self.amount is None:
            raise ValueError(
                f"amount is required for rule_type={self.rule_type!r}"
            )


@dataclass
class SpendRecord:
    """A single spend event recorded in the audit log."""

    agent_id: str
    amount: float
    currency: str
    facilitator_url: str
    tx_hash: Optional[str]
    success: bool
    timestamp: float = field(default_factory=time.time)
    decision: str = "allowed"


@dataclass
class SpendDecision:
    """Result of a policy check before executing a payment."""

    approved: bool
    requires_approval: bool = False
    reason: str = ""
    agent_id: str = ""
    amount: float = 0.0
    currency: str = ""
    budget_remaining: Optional[float] = None


class SpendTracker:
    """
    Tracks cumulative spend per agent within rolling time windows.

    Thread-safe: uses a reentrant lock internally.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # agent_id -> list of (timestamp, amount) tuples
        self._records: dict[str, list[tuple[float, float]]] = defaultdict(list)

    def record(self, agent_id: str, amount: float, timestamp: float | None = None) -> None:
        """Record a spend event for an agent."""
        ts = timestamp if timestamp is not None else time.time()
        with self._lock:
            self._records[agent_id].append((ts, amount))

    def total_in_window(self, agent_id: str, window_seconds: int) -> float:
        """Return cumulative spend for agent within the last *window_seconds*."""
        cutoff = time.time() - window_seconds
        with self._lock:
            records = self._records.get(agent_id, [])
            # Prune old records while we're holding the lock
            pruned = [(ts, amt) for ts, amt in records if ts >= cutoff]
            self._records[agent_id] = pruned
            return sum(amt for _, amt in pruned)

    def reset(self, agent_id: str) -> None:
        """Clear all spend records for an agent (e.g. for testing)."""
        with self._lock:
            self._records[agent_id] = []


class CircuitBreaker:
    """
    Auto-pauses payments to a facilitator when consecutive failures exceed a threshold.

    Once tripped, the circuit remains open (payments blocked) until
    *recovery_timeout_seconds* have elapsed, at which point it moves to
    HALF_OPEN.  A single successful payment resets it to CLOSED.

    Thread-safe.
    """

    class State(str, Enum):
        CLOSED = "closed"      # normal operation
        OPEN = "open"          # payments blocked
        HALF_OPEN = "half_open"  # one test payment allowed

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: int = 30,
    ) -> None:
        self._threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_seconds
        self._lock = threading.RLock()
        # facilitator_url -> state info
        self._states: dict[str, dict] = {}

    def _get_state(self, url: str) -> dict:
        if url not in self._states:
            self._states[url] = {
                "state": self.State.CLOSED,
                "failures": 0,
                "opened_at": None,
            }
        return self._states[url]

    def is_open(self, facilitator_url: str) -> bool:
        """Return True if the circuit is OPEN (payments should be blocked)."""
        with self._lock:
            info = self._get_state(facilitator_url)
            if info["state"] == self.State.OPEN:
                elapsed = time.time() - info["opened_at"]
                if elapsed >= self._recovery_timeout:
                    info["state"] = self.State.HALF_OPEN
                    return False
                return True
            return False

    def record_success(self, facilitator_url: str) -> None:
        """Record a successful settlement — resets failure count and closes circuit."""
        with self._lock:
            info = self._get_state(facilitator_url)
            info["failures"] = 0
            info["state"] = self.State.CLOSED
            info["opened_at"] = None

    def record_failure(self, facilitator_url: str) -> None:
        """Record a failed settlement — may trip the circuit breaker."""
        with self._lock:
            info = self._get_state(facilitator_url)
            info["failures"] += 1
            if info["failures"] >= self._threshold and info["state"] == self.State.CLOSED:
                info["state"] = self.State.OPEN
                info["opened_at"] = time.time()

    def get_state(self, facilitator_url: str) -> "CircuitBreaker.State":
        with self._lock:
            return self._get_state(facilitator_url)["state"]

    def reset(self, facilitator_url: str) -> None:
        """Manually reset the circuit to CLOSED (e.g. after operator intervention)."""
        with self._lock:
            self._states.pop(facilitator_url, None)


class AuditLog:
    """In-memory immutable audit trail of spend decisions and settlements."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: list[SpendRecord] = []

    def append(self, record: SpendRecord) -> None:
        with self._lock:
            self._entries.append(record)

    def records(self, agent_id: str | None = None) -> list[SpendRecord]:
        """Return all records, optionally filtered by agent_id."""
        with self._lock:
            if agent_id is None:
                return list(self._entries)
            return [r for r in self._entries if r.agent_id == agent_id]

    def as_dicts(self, agent_id: str | None = None) -> list[dict]:
        return [
            {
                "agent_id": r.agent_id,
                "amount": r.amount,
                "currency": r.currency,
                "facilitator_url": r.facilitator_url,
                "tx_hash": r.tx_hash,
                "success": r.success,
                "timestamp": datetime.fromtimestamp(r.timestamp, tz=timezone.utc).isoformat(),
                "decision": r.decision,
            }
            for r in self.records(agent_id)
        ]


class PaymentPolicyEngine:
    """
    Central payment control plane for Swarms x402 integrations.

    Combines per-agent budgets, ordered spending rules, circuit breakers,
    and an audit log into a single, thread-safe policy engine.

    Example::

        engine = PaymentPolicyEngine()
        engine.add_budget(AgentBudget("research-bot", max_amount=500, window="daily"))
        engine.add_rule(SpendingRule("block_above", amount=1000, currency="USDC"))
        engine.add_rule(SpendingRule("require_approval_above", amount=100, currency="USDC"))
        engine.add_rule(SpendingRule("allow_all"))

        decision = engine.check_payment("research-bot", 25.0, "USDC", "https://x402.org/facilitator")
        if decision.approved and not decision.requires_approval:
            # execute payment, then:
            engine.record_payment("research-bot", 25.0, "USDC", "https://x402.org/facilitator", "0xabc", True)
    """

    def __init__(
        self,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        self._budgets: dict[str, AgentBudget] = {}
        self._rules: list[SpendingRule] = []
        self._tracker = SpendTracker()
        self._circuit_breaker = circuit_breaker or CircuitBreaker()
        self._audit_log = AuditLog()

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def add_budget(self, budget: AgentBudget) -> None:
        """Register a per-agent spending budget."""
        self._budgets[budget.agent_id] = budget

    def add_rule(self, rule: SpendingRule) -> None:
        """Append a spending rule.  Rules are evaluated in insertion order."""
        self._rules.append(rule)

    def load_policy(self, policy: dict) -> None:
        """
        Load budgets and rules from a dict (mirrors the PaySentry policy schema).

        Expected format::

            {
                "budgets": [
                    {"agent_id": "research-bot", "window": "daily", "max_amount": 500, "currency": "USDC"},
                ],
                "rules": [
                    {"type": "block_above", "amount": 1000, "currency": "USDC"},
                    {"type": "require_approval_above", "amount": 100, "currency": "USDC"},
                    {"type": "allow_all"},
                ],
            }
        """
        for b in policy.get("budgets", []):
            self.add_budget(
                AgentBudget(
                    agent_id=b["agent_id"],
                    max_amount=float(b["max_amount"]),
                    currency=b.get("currency", "USDC"),
                    window=b.get("window", "daily"),
                )
            )
        for r in policy.get("rules", []):
            rule_type = r.get("type") or r.get("rule_type")
            self.add_rule(
                SpendingRule(
                    rule_type=rule_type,
                    amount=r.get("amount"),
                    currency=r.get("currency"),
                )
            )

    # ------------------------------------------------------------------
    # Core policy check
    # ------------------------------------------------------------------

    def check_payment(
        self,
        agent_id: str,
        amount: float,
        currency: str = "USDC",
        facilitator_url: str = "",
    ) -> SpendDecision:
        """
        Evaluate whether a payment should proceed.

        Returns a :class:`SpendDecision`.  Call :meth:`record_payment` after
        the actual settlement completes (success or failure).
        """
        base = SpendDecision(
            approved=False,
            agent_id=agent_id,
            amount=amount,
            currency=currency,
        )

        # 1. Circuit breaker check
        if facilitator_url and self._circuit_breaker.is_open(facilitator_url):
            base.reason = (
                f"Circuit breaker is OPEN for facilitator {facilitator_url!r}. "
                "Too many consecutive failures."
            )
            return base

        # 2. Per-agent budget check
        if agent_id in self._budgets:
            budget = self._budgets[agent_id]
            spent = self._tracker.total_in_window(agent_id, budget.window_seconds())
            remaining = budget.max_amount - spent
            base.budget_remaining = remaining
            if amount > remaining:
                base.reason = (
                    f"Agent {agent_id!r} would exceed {budget.window} budget "
                    f"({budget.max_amount} {budget.currency}). "
                    f"Already spent: {spent:.4f}, requested: {amount:.4f}, remaining: {remaining:.4f}."
                )
                return base

        # 3. Ordered rule evaluation
        for rule in self._rules:
            if rule.rule_type == RuleType.ALLOW_ALL.value:
                base.approved = True
                base.reason = "Allowed by allow_all rule."
                return base

            if rule.currency and rule.currency != currency:
                continue  # rule applies to a different currency

            if rule.rule_type == RuleType.BLOCK_ABOVE.value:
                if amount > rule.amount:
                    base.reason = (
                        f"Payment of {amount} {currency} exceeds block_above threshold "
                        f"of {rule.amount} {currency}."
                    )
                    return base

            elif rule.rule_type == RuleType.REQUIRE_APPROVAL_ABOVE.value:
                if amount > rule.amount:
                    base.approved = True
                    base.requires_approval = True
                    base.reason = (
                        f"Payment of {amount} {currency} exceeds require_approval_above "
                        f"threshold of {rule.amount} {currency}. Human approval required."
                    )
                    return base

        # No rule matched — default allow if rules list is empty
        if not self._rules:
            base.approved = True
            base.reason = "No rules configured — default allow."

        return base

    # ------------------------------------------------------------------
    # Settlement recording
    # ------------------------------------------------------------------

    def record_payment(
        self,
        agent_id: str,
        amount: float,
        currency: str,
        facilitator_url: str,
        tx_hash: Optional[str] = None,
        success: bool = True,
    ) -> SpendRecord:
        """
        Record the outcome of an attempted payment.

        Updates:
        - SpendTracker (on success)
        - CircuitBreaker (success resets, failure increments)
        - AuditLog (always)
        """
        if success:
            self._tracker.record(agent_id, amount)
            if facilitator_url:
                self._circuit_breaker.record_success(facilitator_url)
        else:
            if facilitator_url:
                self._circuit_breaker.record_failure(facilitator_url)

        record = SpendRecord(
            agent_id=agent_id,
            amount=amount,
            currency=currency,
            facilitator_url=facilitator_url,
            tx_hash=tx_hash,
            success=success,
            decision="allowed" if success else "failed",
        )
        self._audit_log.append(record)
        return record

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def audit_log(self) -> AuditLog:
        return self._audit_log

    @property
    def spend_tracker(self) -> SpendTracker:
        return self._tracker

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        return self._circuit_breaker

    def agent_spend_summary(self, agent_id: str) -> dict:
        """Return current spend totals for an agent across all configured windows."""
        summary: dict = {"agent_id": agent_id, "windows": {}}
        if agent_id in self._budgets:
            budget = self._budgets[agent_id]
            spent = self._tracker.total_in_window(agent_id, budget.window_seconds())
            summary["windows"][budget.window] = {
                "spent": spent,
                "limit": budget.max_amount,
                "remaining": budget.max_amount - spent,
                "currency": budget.currency,
            }
        return summary


__all__ = [
    "AgentBudget",
    "SpendingRule",
    "SpendRecord",
    "SpendDecision",
    "SpendTracker",
    "CircuitBreaker",
    "AuditLog",
    "PaymentPolicyEngine",
    "WindowType",
    "RuleType",
]
