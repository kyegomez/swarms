import inspect
from functools import lru_cache
from typing import Literal, Tuple, get_args

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

# Fallback set, unioned with installed litellm's values so supported efforts are never reduced.
REASONING_EFFORTS: Tuple[str, ...] = get_args(ReasoningEffort)


@lru_cache(maxsize=1)
def get_reasoning_efforts() -> Tuple[str, ...]:
    """
    Returns reasoning_effort values from installed litellm, or fallback set.
    """
    values: Tuple[str, ...] = ()

    try:
        import litellm

        annotation = (
            inspect.signature(litellm.completion)
            .parameters["reasoning_effort"]
            .annotation
        )
        values = get_args(get_args(annotation)[0])
    except Exception:
        # Fallback if litellm is missing or signature unexpected.
        values = ()

    # Deduplicate while preserving order.
    return tuple(dict.fromkeys((*values, *REASONING_EFFORTS)))
