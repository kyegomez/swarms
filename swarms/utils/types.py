"""
Type definitions for the swarms package.

This module contains common type definitions used across the swarms package
to avoid circular import issues.
"""

from typing import Literal

# Return types for agent creation functions
ReturnTypes = Literal[
    "auto", "swarm", "agents", "both", "tasks", "run_swarm"
]
