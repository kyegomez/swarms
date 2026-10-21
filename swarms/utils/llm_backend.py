"""The library swarms makes model calls through.

RouteHub is the only model-call backend and a required dependency. The
library is not imported until a name is first used, so importing swarms
stays fast. completion, acompletion and embedding are defined here; every
other name, such as get_model_info or AuthenticationError, is read from
RouteHub on first access.

SWARMS_LLM_BACKEND is no longer read. A leftover "litellm" value draws a
warning so old setups learn the switch is gone; every other value is
ignored.
"""

import importlib
import os
from types import ModuleType
from typing import Any

from loguru import logger

# The library refreshes these itself, so they are read on every access.
_UNCACHED = frozenset({"model_list", "model_cost"})

_module = None


def backend_module() -> ModuleType:
    """Import RouteHub on first use.

    Returns:
        ModuleType: The routehub module.
    """
    global _module
    if _module is None:
        if (
            os.getenv("SWARMS_LLM_BACKEND", "").strip().lower()
            == "litellm"
        ):
            logger.warning(
                "SWARMS_LLM_BACKEND=litellm is no longer supported; "
                "swarms makes model calls through RouteHub."
            )
        _module = importlib.import_module("routehub")
    return _module


def completion(*args: Any, **kwargs: Any) -> Any:
    """Make a chat completion call.

    Args:
        *args (Any): Positional arguments for RouteHub's completion.
        **kwargs (Any): Keyword arguments for RouteHub's completion.

    Returns:
        Any: The response, or a stream of chunks when stream is on.
    """
    return backend_module().completion(*args, **kwargs)


async def acompletion(*args: Any, **kwargs: Any) -> Any:
    """Make an async chat completion call.

    Args:
        *args (Any): Positional arguments for RouteHub's acompletion.
        **kwargs (Any): Keyword arguments for RouteHub's acompletion.

    Returns:
        Any: The response, or an async stream of chunks when stream is on.
    """
    return await backend_module().acompletion(*args, **kwargs)


def embedding(*args: Any, **kwargs: Any) -> Any:
    """Make an embedding call.

    Args:
        *args (Any): Positional arguments for RouteHub's embedding.
        **kwargs (Any): Keyword arguments for RouteHub's embedding.

    Returns:
        Any: The embedding response.
    """
    return backend_module().embedding(*args, **kwargs)


def __getattr__(name: str) -> Any:
    """Read a name from RouteHub on first access.

    Args:
        name (str): A name RouteHub exports, such as get_model_info.

    Returns:
        Any: The library's function, class or value.

    Raises:
        AttributeError: If RouteHub has no such name.
    """
    # Dunder lookups come from tooling such as import machinery, not callers.
    if name.startswith("__"):
        raise AttributeError(name)
    value = getattr(backend_module(), name)
    if name not in _UNCACHED:
        globals()[name] = value
    return value
