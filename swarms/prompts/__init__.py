import importlib
from typing import Any

_MODULE_MAP = {
    "AUTO_AGENT_BUILDER_SYSTEM_PROMPT": "auto_agent_builder_prompt",
    "CODE_INTERPRETER": "code_interpreter",
    "DOCUMENTATION_WRITER_SOP": "documentation",
    "FINANCE_AGENT_PROMPT": "finance_agent_prompt",
    "GROWTH_AGENT_PROMPT": "growth_agent_prompt",
    "LEGAL_AGENT_PROMPT": "legal_agent_prompt",
    "OPERATIONS_AGENT_PROMPT": "operations_agent_prompt",
    "PRODUCT_AGENT_PROMPT": "product_agent_prompt",
    "Prompt": "prompt",
    "autonomous_agent_system_prompt": "autonomous_agent_prompt",
    "get_autonomous_agent_prompt": "autonomous_agent_prompt",
    "get_autonomous_agent_prompt_with_context": "autonomous_agent_prompt",
}

__all__ = [
    "AUTO_AGENT_BUILDER_SYSTEM_PROMPT",
    "CODE_INTERPRETER",
    "DOCUMENTATION_WRITER_SOP",
    "FINANCE_AGENT_PROMPT",
    "GROWTH_AGENT_PROMPT",
    "LEGAL_AGENT_PROMPT",
    "OPERATIONS_AGENT_PROMPT",
    "PRODUCT_AGENT_PROMPT",
    "Prompt",
    "autonomous_agent_system_prompt",
    "get_autonomous_agent_prompt",
    "get_autonomous_agent_prompt_with_context",
]


def __getattr__(name: str) -> Any:
    """Resolve prompt attributes lazily on first access (PEP 562)."""
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
