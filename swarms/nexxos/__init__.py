"""
NexxOS - AI-native Operational Infrastructure Platform

NexxOS is an enterprise-grade operational infrastructure platform built on top of Swarms AI orchestration.
It extends Swarms with governance, persistent organizational memory, observability, economic intelligence,
and operational reliability for autonomous enterprise systems.

Architecture:
- Core: Governance, Economics, Observability, Memory, Security
- Agents: NexxAgent (Swarms Agent wrapper with enterprise features)
- Workflows: Pre-built enterprise workflows (ecommerce, finance, operations, etc.)
- Dashboards: NexxSight observability dashboard

Key Features:
- Governance Engine: Permission control, approval workflows, policy enforcement
- Economic Intelligence: Token tracking, cost analysis, ROI calculation
- Observability: Execution timelines, workflow graphs, anomaly detection, replay
- Persistent Memory: Institutional knowledge, SOPs, organizational patterns
- Security: Zero-trust model, tool sandboxing, credential vault
"""

from swarms.nexxos.core import (
    GovernanceEngine,
    EconomicsEngine,
    ObservabilityEngine,
    MemoryController,
    SecurityEngine,
)
from swarms.nexxos.agents import NexxAgent

__version__ = "0.1.0"
__all__ = [
    "NexxAgent",
    "GovernanceEngine",
    "EconomicsEngine",
    "ObservabilityEngine",
    "MemoryController",
    "SecurityEngine",
]
