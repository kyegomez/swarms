"""RouteHub, the library swarms makes model calls through.

completion, acompletion and embedding are defined here so the framework
and its tests have one place to patch. Every other name, such as
get_model_info or AuthenticationError, is read from routehub on first
access. RouteHub loads its own submodules lazily, so importing swarms
stays fast.
"""

from typing import Any

import routehub

# RouteHub refreshes these itself, so they are read on every access.
_UNCACHED = frozenset({"model_list", "model_cost"})


def completion(*args: Any, **kwargs: Any) -> Any:
    """Make a chat completion call.

    Args:
        *args (Any): Positional arguments for routehub.completion.
        **kwargs (Any): Keyword arguments for routehub.completion.

    Returns:
        Any: The response, or a stream of chunks when stream is on.
    """
    return routehub.completion(*args, **kwargs)


async def acompletion(*args: Any, **kwargs: Any) -> Any:
    """Make an async chat completion call.

    Args:
        *args (Any): Positional arguments for routehub.acompletion.
        **kwargs (Any): Keyword arguments for routehub.acompletion.

    Returns:
        Any: The response, or an async stream of chunks when stream is on.
    """
    return await routehub.acompletion(*args, **kwargs)


def embedding(*args: Any, **kwargs: Any) -> Any:
    """Make an embedding call.

    Args:
        *args (Any): Positional arguments for routehub.embedding.
        **kwargs (Any): Keyword arguments for routehub.embedding.

    Returns:
        Any: The embedding response.
    """
    return routehub.embedding(*args, **kwargs)


def __getattr__(name: str) -> Any:
    """Read a name from routehub on first access.

    Args:
        name (str): A name routehub exports, such as get_model_info.

    Returns:
        Any: RouteHub's function, class or value.

    Raises:
        AttributeError: If routehub has no such name.
    """
    # Dunder lookups come from tooling such as import machinery, not callers.
    if name.startswith("__"):
        raise AttributeError(name)
    value = getattr(routehub, name)
    if name not in _UNCACHED:
        globals()[name] = value
    return value
