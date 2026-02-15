from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from swarms.payments.policy import (
    AuditLogger,
    CircuitBreaker,
    PolicyEngine,
    SpendTracker,
)


class SpendingPolicyError(Exception):
    pass


class CircuitBreakerOpenError(Exception):
    pass


@dataclass
class X402PolicyConfig:
    agent_id: str
    currency: str = "USDC"
    default_window: str = "daily"


class X402PolicyClient:
    def __init__(
        self,
        account: Any,
        base_url: str,
        policy_engine: PolicyEngine,
        spend_tracker: SpendTracker,
        circuit_breaker: Optional[CircuitBreaker] = None,
        audit_logger: Optional[AuditLogger] = None,
        policy_config: Optional[X402PolicyConfig] = None,
    ) -> None:
        self.account = account
        self.base_url = base_url
        self.policy_engine = policy_engine
        self.spend_tracker = spend_tracker
        self.circuit_breaker = circuit_breaker
        self.audit_logger = audit_logger
        self.policy_config = policy_config or X402PolicyConfig(
            agent_id="unknown"
        )
        self._client = None

    async def __aenter__(self) -> "X402PolicyClient":
        try:
            from x402.clients.httpx import x402HttpxClient
        except ImportError as exc:
            raise ImportError(
                "x402 is required for X402PolicyClient. Install with `pip install x402`."
            ) from exc

        self._client = x402HttpxClient(
            account=self.account, base_url=self.base_url
        )
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            await self._client.__aexit__(exc_type, exc, tb)
            self._client = None

    async def get(
        self,
        endpoint: str,
        amount: float,
        currency: Optional[str] = None,
        facilitator: Optional[str] = None,
        window: Optional[str] = None,
        **kwargs,
    ) -> Any:
        return await self._request(
            "get",
            endpoint,
            amount,
            currency=currency,
            facilitator=facilitator,
            window=window,
            **kwargs,
        )

    async def _request(
        self,
        method: str,
        endpoint: str,
        amount: float,
        currency: Optional[str] = None,
        facilitator: Optional[str] = None,
        window: Optional[str] = None,
        **kwargs,
    ) -> Any:
        if self._client is None:
            raise RuntimeError(
                "X402PolicyClient is not initialized. Use 'async with'."
            )

        currency = currency or self.policy_config.currency
        window = window or self.policy_config.default_window
        agent_id = self.policy_config.agent_id
        now = datetime.now(timezone.utc)

        decision = self.policy_engine.evaluate(
            agent_id=agent_id, amount=amount, currency=currency, now=now
        )
        if self.audit_logger is not None:
            self.audit_logger.log(
                {
                    "agent_id": agent_id,
                    "amount": amount,
                    "currency": currency,
                    "decision": decision.status,
                    "reason": decision.reason,
                    "timestamp": now.isoformat(),
                }
            )

        if decision.status != "allowed":
            raise SpendingPolicyError(
                f"policy_{decision.status}: {decision.reason}"
            )

        if self.circuit_breaker and facilitator:
            if not self.circuit_breaker.allow(facilitator, now=now):
                raise CircuitBreakerOpenError(
                    "circuit_breaker_open"
                )

        response = await getattr(self._client, method)(
            endpoint, **kwargs
        )

        if self.circuit_breaker and facilitator:
            if response.status_code >= 500:
                self.circuit_breaker.record_failure(
                    facilitator, now=now
                )
            else:
                self.circuit_breaker.record_success(facilitator)

        if response.status_code < 400:
            self.spend_tracker.record_spend(
                agent_id, amount, currency, window, now
            )

        return response
