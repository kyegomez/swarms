import concurrent.futures
from dotenv import load_dotenv

# from swarms.structs.workspace_manager import WorkspaceManager
# workspace_manager = WorkspaceManager()
# workspace_manager.run()

load_dotenv()

# ruff: noqa: E402 # Ignore module level import not at top of file

from swarms.telemetry.bootup import bootup  # noqa: F403
from swarms.telemetry.sentry_active import (
    activate_sentry,
)  # noqa: E402

# Use ThreadPoolExecutor to run bootup and activate_sentry concurrently
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    executor.submit(bootup)
    executor.submit(activate_sentry)

from swarms.agents import *  # noqa: E402, F403
from swarms.artifacts import *  # noqa: E402, F403
from swarms.prompts import *  # noqa: E402, F403
from swarms.schemas import *  # noqa: E402, F403
from swarms.structs import *  # noqa: E402, F403
from swarms.telemetry import *  # noqa: E402, F403
from swarms.tools import *  # noqa: E402, F403
from swarms.utils import *  # noqa: E402, F403
