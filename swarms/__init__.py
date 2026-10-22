from swarms.env import load_swarms_env

load_swarms_env()

from swarms.telemetry.bootup import bootup  # noqa: E402, F403

bootup()

# Subpackages other than structs/prompts still eager-export for compatibility.
# structs and prompts resolve through ``__getattr__`` so ``from swarms import Agent``
# does not load every structure and prompt module (#2734).
from swarms.agents import *  # noqa: E402, F403
from swarms.schemas import *  # noqa: E402, F403
from swarms.telemetry import *  # noqa: E402, F403
from swarms.tools import *  # noqa: E402, F403
from swarms.utils import *  # noqa: E402, F403

import swarms.prompts as _prompts_pkg  # noqa: E402
import swarms.structs as _structs_pkg  # noqa: E402

_LAZY_PACKAGES = (_structs_pkg, _prompts_pkg)
_LAZY_NAMES = {
    name: pkg for pkg in _LAZY_PACKAGES for name in pkg.__all__
}


def __getattr__(name: str):
    """Resolve ``__version__`` and lazy struct/prompt exports on first access."""
    if name == "__version__":
        from importlib.metadata import (
            PackageNotFoundError,
            version,
        )

        try:
            resolved = version("swarms")
        except PackageNotFoundError:
            # A source checkout with no installed distribution.
            resolved = "unknown"

        globals()["__version__"] = resolved
        return resolved

    pkg = _LAZY_NAMES.get(name)
    if pkg is not None:
        value = getattr(pkg, name)
        globals()[name] = value
        return value

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def __dir__() -> list:
    """Advertise ``__version__`` and lazy exports before anything has accessed them."""
    return sorted(set(globals()) | {"__version__"} | set(_LAZY_NAMES))
