"""The library swarms makes model calls through.

RouteHub is used when it is installed, which pip install "swarms[fast]"
does, and litellm otherwise. Set SWARMS_LLM_BACKEND to "litellm" or
"routehub" to choose one explicitly.

Neither library is imported until a name is first used, so importing swarms
stays fast. completion, acompletion and embedding are defined here; every
other name, such as get_model_info or AuthenticationError, is read from the
chosen library on first access.
"""

import importlib
import importlib.util
import os
from types import ModuleType
from typing import Any

BACKENDS = ("routehub", "litellm")

# The libraries refresh these themselves, so they are read on every access.
_UNCACHED = frozenset({"model_list", "model_cost"})

_INSTALL_HINTS = {
    "routehub": 'pip install "swarms[fast]"',
    "litellm": "pip install litellm",
}


def select_backend() -> str:
    """Name the library to make model calls through.

    Returns:
        str: "routehub" or "litellm".

    Raises:
        ValueError: If SWARMS_LLM_BACKEND names another library.
        ImportError: If SWARMS_LLM_BACKEND names a library that is not
            installed.
    """
    requested = os.getenv("SWARMS_LLM_BACKEND", "").strip().lower()
    if not requested:
        if importlib.util.find_spec("routehub") is not None:
            return "routehub"
        return "litellm"
    if requested not in BACKENDS:
        raise ValueError(
            f"SWARMS_LLM_BACKEND must be one of {', '.join(BACKENDS)}, "
            f"got {requested!r}."
        )
    if importlib.util.find_spec(requested) is None:
        raise ImportError(
            f"SWARMS_LLM_BACKEND is {requested!r}, but {requested} is not "
            f"installed. Install it with: {_INSTALL_HINTS[requested]}"
        )
    return requested


BACKEND = select_backend()

_module = None


def backend_module() -> ModuleType:
    """Import the chosen library on first use.

    Returns:
        ModuleType: The routehub or litellm module.
    """
    global _module
    if _module is None:
        _module = importlib.import_module(BACKEND)
    return _module


def completion(*args: Any, **kwargs: Any) -> Any:
    """Make a chat completion call.

    Args:
        *args (Any): Positional arguments for the library's completion.
        **kwargs (Any): Keyword arguments for the library's completion.

    Returns:
        Any: The response, or a stream of chunks when stream is on.
    """
    return backend_module().completion(*args, **kwargs)


async def acompletion(*args: Any, **kwargs: Any) -> Any:
    """Make an async chat completion call.

    Args:
        *args (Any): Positional arguments for the library's acompletion.
        **kwargs (Any): Keyword arguments for the library's acompletion.

    Returns:
        Any: The response, or an async stream of chunks when stream is on.
    """
    return await backend_module().acompletion(*args, **kwargs)


def embedding(*args: Any, **kwargs: Any) -> Any:
    """Make an embedding call.

    Args:
        *args (Any): Positional arguments for the library's embedding.
        **kwargs (Any): Keyword arguments for the library's embedding.

    Returns:
        Any: The embedding response.
    """
    return backend_module().embedding(*args, **kwargs)


def __getattr__(name: str) -> Any:
    """Read a name from the chosen library on first access.

    Args:
        name (str): A name the library exports, such as get_model_info.

    Returns:
        Any: The library's function, class or value.

    Raises:
        AttributeError: If the library has no such name.
    """
    # Dunder lookups come from tooling such as import machinery, not callers.
    if name.startswith("__"):
        raise AttributeError(name)
    value = getattr(backend_module(), name)
    if name not in _UNCACHED:
        globals()[name] = value
    return value
