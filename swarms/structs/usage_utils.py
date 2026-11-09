from typing import Any

from swarms.structs.agent import Agent
from swarms.utils.litellm_wrapper import empty_usage

_USAGE_KEYS = empty_usage().keys()


def total_usage(*roots: Any) -> dict:
    """
    Sum provider token usage over everything the roots hold.

    Args:
        *roots: Structures, agents, model clients, usage dicts, or
            containers of them. Each distinct agent and usage dict is
            counted once.

    Returns:
        A new dict with input_tokens, output_tokens, cached_tokens,
        reasoning_tokens and total_tokens.
    """
    total = empty_usage()
    seen = set()
    stack = list(roots)
    while stack:
        value = stack.pop()
        if isinstance(value, (str, bytes, int, float, bool)):
            continue
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, Agent):
            usage = value.usage
        elif isinstance(value, dict) and value.keys() == _USAGE_KEYS:
            usage = value
        elif isinstance(value, dict):
            stack.extend(value.values())
            continue
        elif isinstance(value, (list, tuple, set, frozenset)):
            stack.extend(value)
            continue
        elif hasattr(value, "__dict__") and any(
            cls.__module__.startswith("swarms.")
            for cls in type(value).__mro__
        ):
            stack.extend(vars(value).values())
            continue
        else:
            continue
        for key in total:
            total[key] += usage.get(key, 0)
    return total
