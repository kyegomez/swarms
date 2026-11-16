from typing import Any

from swarms.utils.litellm_wrapper import empty_usage

_USAGE_KEYS = empty_usage().keys()


def _is_usage(value: Any) -> bool:
    return isinstance(value, dict) and value.keys() == _USAGE_KEYS


def total_usage(*roots: Any) -> dict:
    """
    Sum provider token usage over everything the roots hold.

    Each distinct object is visited once, under three rules:

    - An object with a usage property that keeps its running
      total in a _usage dict is read through usage and not
      searched further.
    - A usage dict is counted when it is a root, sits in a root
      container, or is a direct attribute of an object.
    - A usage dict inside a dict or list that an object holds is a
      stored copy of a total and is skipped. The other values there,
      such as agents, are still searched.

    Args:
        *roots: Structures, agents, model clients, usage dicts, or
            containers of them.

    Returns:
        A new dict with input_tokens, output_tokens, cached_tokens,
        reasoning_tokens and total_tokens.
    """
    total = empty_usage()
    seen = set()
    stack = [(root, True) for root in roots]
    while stack:
        value, counted = stack.pop()
        if isinstance(value, (str, bytes, int, float, bool)):
            continue
        if id(value) in seen or (_is_usage(value) and not counted):
            continue
        seen.add(id(value))
        if _is_usage(value):
            usage = value
        elif isinstance(value, dict):
            stack.extend((item, counted) for item in value.values())
            continue
        elif isinstance(value, (list, tuple, set, frozenset)):
            stack.extend((item, counted) for item in value)
            continue
        elif hasattr(value, "__dict__") and any(
            cls.__module__.startswith("swarms.")
            for cls in type(value).__mro__
        ):
            attributes = vars(value)
            if not (
                isinstance(attributes.get("_usage"), dict)
                and hasattr(type(value), "usage")
            ):
                stack.extend(
                    (item, _is_usage(item))
                    for item in attributes.values()
                )
                continue
            usage = value.usage
        else:
            continue
        for key in total:
            total[key] += usage.get(key, 0)
    return total
