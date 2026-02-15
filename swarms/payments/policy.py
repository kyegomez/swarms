from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class PolicyRule:
    type: str
    amount: Optional[float] = None
    currency: Optional[str] = None
    rule_id: Optional[str] = None


@dataclass
class Budget:
    agent_id: str
    window: str
    max_amount: float
    currency: str


@dataclass
class PolicyDecision:
    status: str
    reason: str
    rule_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AuditLogger:
    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    def log(self, entry: Dict[str, Any]) -> None:
        self.entries.append(entry)


class SpendTracker:
    def __init__(self) -> None:
        self._spend: Dict[str, float] = {}

    def _window_key(
        self,
        agent_id: str,
        currency: str,
        window: str,
        now: datetime,
    ) -> str:
        if window == "hourly":
            bucket = now.strftime("%Y-%m-%dT%H")
        elif window == "monthly":
            bucket = now.strftime("%Y-%m")
        else:
            bucket = now.strftime("%Y-%m-%d")
        return f"{agent_id}:{currency}:{window}:{bucket}"

    def get_spend(
        self,
        agent_id: str,
        currency: str,
        window: str,
        now: Optional[datetime] = None,
    ) -> float:
        now = now or datetime.now(timezone.utc)
        key = self._window_key(agent_id, currency, window, now)
        return self._spend.get(key, 0.0)

    def record_spend(
        self,
        agent_id: str,
        amount: float,
        currency: str,
        window: str = "daily",
        now: Optional[datetime] = None,
    ) -> float:
        now = now or datetime.now(timezone.utc)
        key = self._window_key(agent_id, currency, window, now)
        self._spend[key] = self._spend.get(key, 0.0) + float(amount)
        return self._spend[key]


class PolicyEngine:
    def __init__(
        self,
        rules: Optional[List[PolicyRule]] = None,
        budgets: Optional[List[Budget]] = None,
        spend_tracker: Optional[SpendTracker] = None,
    ) -> None:
        self.rules = rules or []
        self.budgets = budgets or []
        self.spend_tracker = spend_tracker or SpendTracker()

    def evaluate(
        self,
        agent_id: str,
        amount: float,
        currency: str,
        now: Optional[datetime] = None,
    ) -> PolicyDecision:
        now = now or datetime.now(timezone.utc)

        for budget in self.budgets:
            if budget.agent_id != agent_id:
                continue
            if budget.currency != currency:
                continue
            spent = self.spend_tracker.get_spend(
                agent_id, currency, budget.window, now
            )
            if spent + amount > budget.max_amount:
                return PolicyDecision(
                    status="blocked",
                    reason="budget_exceeded",
                    metadata={
                        "budget_max": budget.max_amount,
                        "budget_window": budget.window,
                        "spent": spent,
                        "attempt": amount,
                    },
                )

        for rule in self.rules:
            if rule.currency and rule.currency != currency:
                continue
            if rule.type == "block_above" and rule.amount is not None:
                if amount > rule.amount:
                    return PolicyDecision(
                        status="blocked",
                        reason="rule_block_above",
                        rule_id=rule.rule_id,
                        metadata={"rule_amount": rule.amount},
                    )
            if (
                rule.type == "require_approval_above"
                and rule.amount is not None
            ):
                if amount > rule.amount:
                    return PolicyDecision(
                        status="requires_approval",
                        reason="rule_require_approval",
                        rule_id=rule.rule_id,
                        metadata={"rule_amount": rule.amount},
                    )
            if rule.type == "allow_all":
                return PolicyDecision(
                    status="allowed",
                    reason="rule_allow_all",
                    rule_id=rule.rule_id,
                )

        return PolicyDecision(status="allowed", reason="default_allow")

    def record_spend(
        self,
        agent_id: str,
        amount: float,
        currency: str,
        window: str = "daily",
        now: Optional[datetime] = None,
    ) -> float:
        return self.spend_tracker.record_spend(
            agent_id, amount, currency, window, now
        )


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: int = 30,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self._state: Dict[str, Dict[str, Any]] = {}

    def allow(
        self, endpoint: str, now: Optional[datetime] = None
    ) -> bool:
        now = now or datetime.now(timezone.utc)
        state = self._state.get(endpoint)
        if not state:
            return True
        opened_at = state.get("opened_at")
        if opened_at is None:
            return True
        elapsed = (now - opened_at).total_seconds()
        if elapsed >= self.recovery_timeout_seconds:
            self._state[endpoint] = {
                "failures": 0,
                "opened_at": None,
            }
            return True
        return False

    def record_failure(
        self, endpoint: str, now: Optional[datetime] = None
    ) -> None:
        now = now or datetime.now(timezone.utc)
        state = self._state.setdefault(
            endpoint, {"failures": 0, "opened_at": None}
        )
        state["failures"] += 1
        if state["failures"] >= self.failure_threshold:
            state["opened_at"] = now

    def record_success(self, endpoint: str) -> None:
        self._state[endpoint] = {"failures": 0, "opened_at": None}
