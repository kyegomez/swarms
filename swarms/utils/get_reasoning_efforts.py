from functools import lru_cache
from typing import Literal, Tuple, get_args

import routehub

ReasoningEffort = Literal[
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "ultra",
    "max",
    "None",
]

REASONING_EFFORTS: Tuple[str, ...] = get_args(ReasoningEffort)


@lru_cache(maxsize=1)
def get_reasoning_efforts() -> Tuple[str, ...]:
    """The reasoning_effort values RouteHub accepts, plus swarms' own.

    Returns:
        Tuple[str, ...]: The values in order, each listed once.
    """
    try:
        values = tuple(routehub.get_reasoning_efforts())
    except Exception:
        values = ()
    return tuple(dict.fromkeys((*values, *REASONING_EFFORTS)))
