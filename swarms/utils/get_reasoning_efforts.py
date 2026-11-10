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

REASONING_EFFORTS: Tuple[str, ...] = get_args(ReasoningEffort)


@lru_cache(maxsize=1)
def get_reasoning_efforts() -> Tuple[str, ...]:
    """
    Returns reasoning_effort values from the installed LLM backend, or fallback set.
    """
    values: Tuple[str, ...] = ()

    try:
        from swarms.utils import llm_backend

        annotation = (
            inspect.signature(llm_backend.backend_module().completion)
            .parameters["reasoning_effort"]
            .annotation
        )
        values = get_args(get_args(annotation)[0])
    except Exception:
        # Fallback if the backend is missing or its signature is unexpected.
        values = ()

    # Deduplicate while preserving order.
    return tuple(dict.fromkeys((*values, *REASONING_EFFORTS)))
