import importlib
from typing import Any

_MODULE_MAP = {
    "AgentJudge": "agent_judge",
    "AgentMarketplaceHandler": "agent_marketplace_handler",
    "ContextCompressor": "context_compressor",
    "GKPAgent": "gkp_agent",
    "IterativeReflectiveExpansion": "i_agent",
    "LLMManager": "llm_manager",
    "ReasoningAgentRouter": "reasoning_agent_router",
    "ReasoningDuo": "reasoning_duo",
    "ReflexionAgent": "flexion_agent",
    "SelfConsistencyAgent": "consistency_agent",
    "SkillsManager": "skills_manager",
    "ThoughtNode": "tree_of_thoughts",
    "TreeOfThoughts": "tree_of_thoughts",
    "TreeOfThoughtsResult": "tree_of_thoughts",
    "agent_types": "reasoning_agent_router",
}

__all__ = [
    "AgentJudge",
    "AgentMarketplaceHandler",
    "ContextCompressor",
    "GKPAgent",
    "IterativeReflectiveExpansion",
    "LLMManager",
    "ReasoningAgentRouter",
    "ReasoningDuo",
    "ReflexionAgent",
    "SelfConsistencyAgent",
    "SkillsManager",
    "ThoughtNode",
    "TreeOfThoughts",
    "TreeOfThoughtsResult",
    "agent_types",
]


def __getattr__(name: str) -> Any:
    """Resolve agent attributes lazily on first access (PEP 562)."""
    if name in _MODULE_MAP:
        module = importlib.import_module(
            f".{_MODULE_MAP[name]}", __name__
        )
        attr = getattr(module, name)
        globals()[name] = attr
        return attr

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def __dir__() -> list:
    """Advertise all public exports before anything has accessed them."""
    return sorted(set(globals()) | set(__all__))
