import os
import concurrent.futures
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Disable logging by default
if os.getenv("SWARMS_VERBOSE_GLOBAL", "False").lower() == "false":
    logger.disable("")

# Import telemetry functions with error handling
from swarms.telemetry.bootup import bootup  # noqa: E402, F403
from swarms.telemetry.sentry_active import (  # noqa: E402
    activate_sentry,
)  # noqa: E402


# Run telemetry functions concurrently with error handling
def run_telemetry():
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=2
        ) as executor:
            future_bootup = executor.submit(bootup)
            future_sentry = executor.submit(activate_sentry)

            # Wait for completion and check for exceptions
            future_bootup.result()
            future_sentry.result()
    except Exception as e:
        logger.error(f"Error running telemetry functions: {e}")


run_telemetry()

from swarms.agents import *  # noqa: E402, F403
from swarms.artifacts import *  # noqa: E402, F403
from swarms.prompts import *  # noqa: E402, F403
from swarms.schemas import *  # noqa: E402, F403
from swarms.structs import *  # noqa: E402, F403
from swarms.telemetry import *  # noqa: E402, F403
from swarms.tools import *  # noqa: E402, F403
from swarms.utils import *  # noqa: E402, F403
