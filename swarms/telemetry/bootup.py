import os
import warnings
from swarms.utils.disable_logging import disable_logging
from swarms.utils.workspace_manager import ensure_workspace_env
from loguru import logger


def bootup():
    """Super-fast initialization of swarms environment"""
    try:
        # Cache env vars
        verbose = os.getenv("SWARMS_VERBOSE_GLOBAL", "False").lower()

        # Configure logging early
        if verbose == "false":
            logger.disable("CRITICAL")
        else:
            logger.enable("")

        # Silence wandb
        os.environ["WANDB_SILENT"] = "true"

        # Only default it, assigning unconditionally discarded the caller's value
        ensure_workspace_env()

        # Suppress deprecation warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Run lightweight telemetry
        try:
            disable_logging()
        except Exception as e:
            logger.debug(f"Telemetry error: {e}")

    except Exception as e:
        logger.debug(f"Bootup error: {str(e)}")
