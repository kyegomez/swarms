"""Compatibility shim for renamed module.

Use swarms.prompts.hierarchical_system_prompt instead.
"""

from warnings import warn

from swarms.prompts.hierarchical_system_prompt import *  # noqa: F401,F403

warn(
    "swarms.prompts.hiearchical_system_prompt is deprecated; use swarms.prompts.hierarchical_system_prompt",
    DeprecationWarning,
    stacklevel=2,
)
