"""Prompt exports, loaded on first access (PEP 562)."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, str] = {
    "AUTO_AGENT_BUILDER_SYSTEM_PROMPT": "swarms.prompts.auto_agent_builder_prompt",
    "CODE_INTERPRETER": "swarms.prompts.code_interpreter",
    "DOCUMENTATION_WRITER_SOP": "swarms.prompts.documentation",
    "FINANCE_AGENT_PROMPT": "swarms.prompts.finance_agent_prompt",
    "GROWTH_AGENT_PROMPT": "swarms.prompts.growth_agent_prompt",
    "LEGAL_AGENT_PROMPT": "swarms.prompts.legal_agent_prompt",
    "OPERATIONS_AGENT_PROMPT": "swarms.prompts.operations_agent_prompt",
    "PRODUCT_AGENT_PROMPT": "swarms.prompts.product_agent_prompt",
    "Prompt": "swarms.prompts.prompt",
    "autonomous_agent_system_prompt": "swarms.prompts.autonomous_agent_prompt",
    "get_autonomous_agent_prompt": "swarms.prompts.autonomous_agent_prompt",
    "get_autonomous_agent_prompt_with_context": "swarms.prompts.autonomous_agent_prompt",
}

__all__ = [
    "AUTO_AGENT_BUILDER_SYSTEM_PROMPT",
    "CODE_INTERPRETER",
    "FINANCE_AGENT_PROMPT",
    "GROWTH_AGENT_PROMPT",
    "LEGAL_AGENT_PROMPT",
    "OPERATIONS_AGENT_PROMPT",
    "PRODUCT_AGENT_PROMPT",
    "DOCUMENTATION_WRITER_SOP",
    "Prompt",
    "autonomous_agent_system_prompt",
    "get_autonomous_agent_prompt",
    "get_autonomous_agent_prompt_with_context",
]


def __getattr__(name: str) -> Any:
    """Import a prompt module the first time one of its exports is read."""
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

