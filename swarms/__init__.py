from swarms.env import load_swarms_env

load_swarms_env()

from swarms.telemetry.bootup import bootup  # noqa: E402, F403

bootup()

from swarms.agents import *  # noqa: E402, F403
from swarms.artifacts import *  # noqa: E402, F403
from swarms.prompts import *  # noqa: E402, F403
from swarms.schemas import *  # noqa: E402, F403
from swarms.structs import *  # noqa: E402, F403
from swarms.telemetry import *  # noqa: E402, F403
from swarms.tools import *  # noqa: E402, F403
from swarms.utils import *  # noqa: E402, F403


# Reachable as `swarms.MCPManager` as before, but resolved on access so the mcp
# package is not imported until something actually uses it.
_LAZY_TOOLS = frozenset({"MCPManager", "MCPFileTokenStorage"})


def __getattr__(name: str):
    if name in _LAZY_TOOLS:
        from swarms import tools

        return getattr(tools, name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
