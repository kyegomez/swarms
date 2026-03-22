"""Tests for swarms.utils.x402_spending_limits."""

import importlib.util
import sys
import time
from pathlib import Path

import pytest

# Import the module directly to avoid pulling in the full swarms package
# and its heavyweight dependencies (psutil, loguru, litellm, etc.)
_module_path = Path(__file__).parents[2] / "swarms" / "utils" / "x402_spending_limits.py"
_spec = importlib.util.spec_from_file_location("x402_spending_limits", _module_path)
_mod = importlib.util.module_from_spec(_spec)
# Register in sys.modules BEFORE exec so dataclass __module__ lookups succeed
sys.modules["x402_spending_limits"] = _mod
_spec.loader.exec_module(_mod)

AgentBudget = _mod.AgentBudget
AuditLog = _mod.AuditLog
CircuitBreaker = _mod.CircuitBreaker
PaymentPolicyEngine = _mod.PaymentPolicyEngine
SpendDecision = _mod.SpendDecision
SpendRecord = _mod.SpendRecord
SpendTracker = _mod.SpendTracker
SpendingRule = _mod.SpendingRule


# ---------------------------------------------------------------------------
# AgentBudget
# ---------------------------------------------------------------------------


class TestAgentBudget:
    def test_defaults(self):
        b = AgentBudget(agent_id="bot", max_amount=100.0)
        assert b.currency == "USDC"
        assert b.window == "daily"

    def test_window_seconds_daily(self):
        b = AgentBudget(agent_id="bot", max_amount=100.0, window="daily")
        assert b.window_seconds() == 86400

    def test_window_seconds_hourly(self):
        b = AgentBudget(agent_id="bot", max_amount=100.0, window="hourly")
        assert b.window_seconds() == 3600

    def test_window_seconds_weekly(self):
        b = AgentBudget(agent_id="bot", max_amount=100.0, window="weekly")
        assert b.window_seconds() == 604800

    def test_window_seconds_monthly(self):
        b = AgentBudget(agent_id="bot", max_amount=100.0, window="monthly")
        assert b.window_seconds() == 2592000


# ---------------------------------------------------------------------------
# SpendingRule
# ---------------------------------------------------------------------------


class TestSpendingRule:
    def test_allow_all(self):
        r = SpendingRule(rule_type="allow_all")
        assert r.amount is None

    def test_block_above_requires_amount(self):
        with pytest.raises(ValueError):
            SpendingRule(rule_type="block_above")

    def test_require_approval_requires_amount(self):
        with pytest.raises(ValueError):
            SpendingRule(rule_type="require_approval_above")

    def test_invalid_rule_type(self):
        with pytest.raises(ValueError):
            SpendingRule(rule_type="unknown_rule")

    def test_valid_block_above(self):
        r = SpendingRule(rule_type="block_above", amount=500.0, currency="USDC")
        assert r.amount == 500.0


# ---------------------------------------------------------------------------
# SpendTracker
# ---------------------------------------------------------------------------


class TestSpendTracker:
    def test_empty_tracker(self):
        tracker = SpendTracker()
        assert tracker.total_in_window("bot", 86400) == 0.0

    def test_single_record(self):
        tracker = SpendTracker()
        tracker.record("bot", 50.0)
        total = tracker.total_in_window("bot", 86400)
        assert total == pytest.approx(50.0)

    def test_multiple_records(self):
        tracker = SpendTracker()
        tracker.record("bot", 10.0)
        tracker.record("bot", 20.0)
        tracker.record("bot", 30.0)
        assert tracker.total_in_window("bot", 86400) == pytest.approx(60.0)

    def test_window_excludes_old_records(self):
        tracker = SpendTracker()
        old_ts = time.time() - 7200  # 2 hours ago
        tracker.record("bot", 100.0, timestamp=old_ts)
        tracker.record("bot", 25.0)  # recent
        # 1-hour window should only include the recent record
        total = tracker.total_in_window("bot", 3600)
        assert total == pytest.approx(25.0)

    def test_reset_clears_records(self):
        tracker = SpendTracker()
        tracker.record("bot", 100.0)
        tracker.reset("bot")
        assert tracker.total_in_window("bot", 86400) == 0.0

    def test_separate_agents(self):
        tracker = SpendTracker()
        tracker.record("agent-a", 50.0)
        tracker.record("agent-b", 30.0)
        assert tracker.total_in_window("agent-a", 86400) == pytest.approx(50.0)
        assert tracker.total_in_window("agent-b", 86400) == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    URL = "https://x402.org/facilitator"

    def test_initially_closed(self):
        cb = CircuitBreaker()
        assert not cb.is_open(self.URL)
        assert cb.get_state(self.URL) == CircuitBreaker.State.CLOSED

    def test_trips_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure(self.URL)
        assert cb.is_open(self.URL)
        assert cb.get_state(self.URL) == CircuitBreaker.State.OPEN

    def test_does_not_trip_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=5)
        for _ in range(4):
            cb.record_failure(self.URL)
        assert not cb.is_open(self.URL)

    def test_success_resets_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(2):
            cb.record_failure(self.URL)
        cb.record_success(self.URL)
        cb.record_failure(self.URL)  # only 1 failure after reset
        assert not cb.is_open(self.URL)

    def test_success_closes_open_circuit(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure(self.URL)
        assert cb.is_open(self.URL)
        cb.record_success(self.URL)
        assert not cb.is_open(self.URL)
        assert cb.get_state(self.URL) == CircuitBreaker.State.CLOSED

    def test_recovery_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=1)
        for _ in range(3):
            cb.record_failure(self.URL)
        assert cb.is_open(self.URL)
        time.sleep(1.1)
        # After timeout, circuit moves to HALF_OPEN and is_open returns False
        assert not cb.is_open(self.URL)
        assert cb.get_state(self.URL) == CircuitBreaker.State.HALF_OPEN

    def test_manual_reset(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure(self.URL)
        cb.reset(self.URL)
        assert not cb.is_open(self.URL)
        assert cb.get_state(self.URL) == CircuitBreaker.State.CLOSED


# ---------------------------------------------------------------------------
# AuditLog
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_empty(self):
        log = AuditLog()
        assert log.records() == []

    def test_append_and_retrieve(self):
        log = AuditLog()
        record = SpendRecord(
            agent_id="bot",
            amount=25.0,
            currency="USDC",
            facilitator_url="https://x402.org/facilitator",
            tx_hash="0xabc",
            success=True,
        )
        log.append(record)
        assert len(log.records()) == 1
        assert log.records()[0].agent_id == "bot"

    def test_filter_by_agent(self):
        log = AuditLog()
        log.append(SpendRecord("agent-a", 10.0, "USDC", "", None, True))
        log.append(SpendRecord("agent-b", 20.0, "USDC", "", None, True))
        log.append(SpendRecord("agent-a", 30.0, "USDC", "", None, True))
        assert len(log.records("agent-a")) == 2
        assert len(log.records("agent-b")) == 1

    def test_as_dicts(self):
        log = AuditLog()
        log.append(SpendRecord("bot", 5.0, "USDC", "https://x402.org", "0x1", True))
        dicts = log.as_dicts()
        assert len(dicts) == 1
        d = dicts[0]
        assert d["agent_id"] == "bot"
        assert d["amount"] == 5.0
        assert "timestamp" in d


# ---------------------------------------------------------------------------
# PaymentPolicyEngine — integration tests
# ---------------------------------------------------------------------------


class TestPaymentPolicyEngine:
    def _engine_with_defaults(self) -> PaymentPolicyEngine:
        engine = PaymentPolicyEngine()
        engine.add_budget(AgentBudget("research-bot", max_amount=500.0, window="daily"))
        engine.add_rule(SpendingRule("block_above", amount=1000.0, currency="USDC"))
        engine.add_rule(SpendingRule("require_approval_above", amount=100.0, currency="USDC"))
        engine.add_rule(SpendingRule("allow_all"))
        return engine

    # --- allow ---

    def test_normal_payment_approved(self):
        engine = self._engine_with_defaults()
        decision = engine.check_payment("research-bot", 25.0)
        assert decision.approved
        assert not decision.requires_approval

    def test_no_rules_default_allow(self):
        engine = PaymentPolicyEngine()
        decision = engine.check_payment("any-bot", 10.0)
        assert decision.approved

    # --- block_above ---

    def test_block_above_threshold(self):
        # Use an agent with no budget so only the rules are evaluated
        engine = self._engine_with_defaults()
        decision = engine.check_payment("no-budget-bot", 1500.0)
        assert not decision.approved
        assert "block_above" in decision.reason

    def test_exactly_at_block_threshold_is_allowed(self):
        engine = self._engine_with_defaults()
        # Exactly 1000 — not *above* 1000; use an agent with no budget
        decision = engine.check_payment("no-budget-bot", 1000.0)
        # 1000 is NOT above 1000, so should hit require_approval_above (>100) first
        assert decision.approved
        assert decision.requires_approval

    # --- require_approval_above ---

    def test_require_approval_above_threshold(self):
        engine = self._engine_with_defaults()
        decision = engine.check_payment("research-bot", 150.0)
        assert decision.approved
        assert decision.requires_approval
        assert "require_approval_above" in decision.reason

    # --- budget exhaustion ---

    def test_budget_blocks_overspend(self):
        engine = self._engine_with_defaults()
        # Record 480 already spent
        engine.spend_tracker.record("research-bot", 480.0)
        # Now try to spend 30 more (480+30=510 > 500 budget)
        decision = engine.check_payment("research-bot", 30.0)
        assert not decision.approved
        assert "budget" in decision.reason.lower()

    def test_budget_allows_within_limit(self):
        engine = self._engine_with_defaults()
        engine.spend_tracker.record("research-bot", 400.0)
        decision = engine.check_payment("research-bot", 50.0)
        assert decision.approved

    def test_budget_remaining_reported(self):
        engine = self._engine_with_defaults()
        engine.spend_tracker.record("research-bot", 300.0)
        decision = engine.check_payment("research-bot", 50.0)
        assert decision.budget_remaining == pytest.approx(200.0)

    def test_unknown_agent_no_budget_check(self):
        engine = self._engine_with_defaults()
        # Agent with no budget configured — only rules apply
        decision = engine.check_payment("unknown-bot", 25.0)
        assert decision.approved

    # --- circuit breaker integration ---

    def test_open_circuit_blocks_payment(self):
        engine = self._engine_with_defaults()
        url = "https://x402.org/facilitator"
        for _ in range(5):
            engine.circuit_breaker.record_failure(url)
        decision = engine.check_payment("research-bot", 10.0, facilitator_url=url)
        assert not decision.approved
        assert "Circuit breaker" in decision.reason

    def test_no_facilitator_url_skips_circuit_check(self):
        engine = self._engine_with_defaults()
        url = "https://x402.org/facilitator"
        for _ in range(5):
            engine.circuit_breaker.record_failure(url)
        # No facilitator URL → circuit breaker not consulted
        decision = engine.check_payment("research-bot", 10.0, facilitator_url="")
        assert decision.approved

    # --- record_payment ---

    def test_record_success_updates_tracker(self):
        engine = self._engine_with_defaults()
        engine.record_payment("research-bot", 50.0, "USDC", "", "0x1", success=True)
        total = engine.spend_tracker.total_in_window("research-bot", 86400)
        assert total == pytest.approx(50.0)

    def test_record_failure_does_not_update_tracker(self):
        engine = self._engine_with_defaults()
        engine.record_payment("research-bot", 50.0, "USDC", "", "0x1", success=False)
        total = engine.spend_tracker.total_in_window("research-bot", 86400)
        assert total == pytest.approx(0.0)

    def test_record_failure_trips_circuit_breaker(self):
        engine = PaymentPolicyEngine(circuit_breaker=CircuitBreaker(failure_threshold=3))
        url = "https://x402.org/facilitator"
        for _ in range(3):
            engine.record_payment("bot", 10.0, "USDC", url, success=False)
        assert engine.circuit_breaker.is_open(url)

    def test_record_success_adds_to_audit_log(self):
        engine = self._engine_with_defaults()
        engine.record_payment("research-bot", 25.0, "USDC", "", "0xabc", True)
        records = engine.audit_log.records("research-bot")
        assert len(records) == 1
        assert records[0].tx_hash == "0xabc"

    # --- load_policy ---

    def test_load_policy_dict(self):
        engine = PaymentPolicyEngine()
        engine.load_policy({
            "budgets": [
                {"agent_id": "bot", "window": "daily", "max_amount": 200, "currency": "USDC"},
            ],
            "rules": [
                {"type": "block_above", "amount": 500, "currency": "USDC"},
                {"type": "allow_all"},
            ],
        })
        decision = engine.check_payment("bot", 100.0)
        assert decision.approved
        decision_blocked = engine.check_payment("bot", 600.0)
        assert not decision_blocked.approved

    # --- agent_spend_summary ---

    def test_agent_spend_summary(self):
        engine = self._engine_with_defaults()
        engine.spend_tracker.record("research-bot", 200.0)
        summary = engine.agent_spend_summary("research-bot")
        assert summary["agent_id"] == "research-bot"
        assert "daily" in summary["windows"]
        info = summary["windows"]["daily"]
        assert info["spent"] == pytest.approx(200.0)
        assert info["limit"] == 500.0
        assert info["remaining"] == pytest.approx(300.0)

    def test_agent_spend_summary_no_budget(self):
        engine = self._engine_with_defaults()
        summary = engine.agent_spend_summary("no-budget-bot")
        assert summary["windows"] == {}

    # --- currency filtering ---

    def test_block_above_different_currency_ignored(self):
        engine = PaymentPolicyEngine()
        engine.add_rule(SpendingRule("block_above", amount=10.0, currency="ETH"))
        engine.add_rule(SpendingRule("allow_all"))
        # Payment in USDC should not be blocked by the ETH rule
        decision = engine.check_payment("bot", 50.0, currency="USDC")
        assert decision.approved
