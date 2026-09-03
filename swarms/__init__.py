from swarms.env import load_swarms_env

load_swarms_env()

from swarms.telemetry.bootup import bootup  # noqa: E402, F403

bootup()

from swarms.agents import *  # noqa: E402, F403
from swarms.prompts import *  # noqa: E402, F403
from swarms.schemas import *  # noqa: E402, F403
from swarms.structs import *  # noqa: E402, F403
from swarms.telemetry import *  # noqa: E402, F403
from swarms.tools import *  # noqa: E402, F403
from swarms.utils import *  # noqa: E402, F403


def __getattr__(name: str) -> str:
    """Resolve ``swarms.__version__`` on first access.

    The version comes from the installed distribution's metadata rather
    than a literal here, so it cannot drift from ``pyproject.toml``.
    Resolving it lazily keeps the dist-info scan off the import path for
    the majority of programs, which never read it.
    """
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

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def __dir__() -> list:
    """Advertise ``__version__`` before anything has accessed it."""
    return sorted(set(globals()) | {"__version__"})
