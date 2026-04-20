"""Compatibility shim for renamed module.

Use swarms.structs.hybrid_hierarchical_peer_swarm instead.
"""

from warnings import warn

from swarms.structs.hybrid_hierarchical_peer_swarm import *  # noqa: F401,F403

warn(
    "swarms.structs.hybrid_hiearchical_peer_swarm is deprecated; use swarms.structs.hybrid_hierarchical_peer_swarm",
    DeprecationWarning,
    stacklevel=2,
)
