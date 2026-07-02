"""
FuguAgent: A multi-agent orchestration system that behaves like a single model API.

This module implements the Fugu/Trinity pattern where a dedicated coordinator
model dynamically assigns tasks to a ranked pool of worker agents using tool-calling.
"""

from examples.multi_agent.fugu_agent.fugu_agent import (
    FuguAgent,
    AgentTask,
    AgentTaskResult,
    VerificationResult,
    WorkflowState,
    MemoryStore,
    MODEL_TIER,
    _model_tier,
    _detect_models,
    _rank_workers,
    _make_decide_tool,
    _build_coordinator_system_prompt,
)

__all__ = [
    "FuguAgent",
    "AgentTask",
    "AgentTaskResult",
    "VerificationResult",
    "WorkflowState",
    "MemoryStore",
    "MODEL_TIER",
    "_model_tier",
    "_detect_models",
    "_rank_workers",
    "_make_decide_tool",
    "_build_coordinator_system_prompt",
]
