import inspect
from functools import lru_cache
from typing import Literal, Tuple, get_args

# Spelled out because a Literal needs literal members: `Literal[SOME_TUPLE]`
# is rejected by type checkers even though it works at runtime.
ReasoningEffort = Literal[
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "ultra",
    "max",
]

# Unioned with whatever the installed litellm advertises, so a downgrade never
# narrows the values Agent already accepts.
REASONING_EFFORTS: Tuple[str, ...] = get_args(ReasoningEffort)


@lru_cache(maxsize=1)
def get_reasoning_efforts() -> Tuple[str, ...]:
    """Mirror the reasoning_effort values accepted by the installed litellm.

    Derived by introspecting ``litellm.completion``'s signature so the schema
    always tracks the pinned litellm version (newer litellm adds levels such
    as 'xhigh' and 'none') instead of hardcoding today's set. Falls back to
    the litellm 1.76.x set if introspection fails.

    Returns:
        Tuple[str, ...]: Accepted reasoning effort levels, in a stable order and
        deduplicated. A tuple is required by callers that splat it into
        ``Literal[...]`` — a list would collapse into a single Literal member.
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
        # litellm missing, or its signature changed shape; the fallback set
        # below is still correct, so this is not worth surfacing to the user.
        values = ()

    # dict.fromkeys dedupes while preserving first-seen order, so the Literal
    # members stay stable regardless of the installed litellm version.
    return tuple(dict.fromkeys((*values, *REASONING_EFFORTS)))
