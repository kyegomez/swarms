"""Compatibility shim for renamed module.

Use swarms.structs.hierarchical_swarm instead.
"""

from warnings import warn

from swarms.structs.hierarchical_swarm import *  # noqa: F401,F403

warn(
    "swarms.structs.hiearchical_swarm is deprecated; use swarms.structs.hierarchical_swarm",
    DeprecationWarning,
    stacklevel=2,
)
