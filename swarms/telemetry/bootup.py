import os
import warnings
from pathlib import Path
from swarms.utils.disable_logging import disable_logging
from loguru import logger


def bootup():
    """Super-fast initialization of swarms environment"""
    try:
        # Cache env vars
        verbose = os.getenv("SWARMS_VERBOSE_GLOBAL", "False").lower()
        workspace_path = Path.cwd() / "agent_workspace"

        # Configure logging early
        if verbose == "false":
            logger.disable("CRITICAL")
        else:
            logger.enable("")

        # Silence wandb
        os.environ["WANDB_SILENT"] = "true"

        # Setup workspace dir only if needed
        if not workspace_path.exists():
            workspace_path.mkdir(parents=True, exist_ok=True)
        os.environ["WORKSPACE_DIR"] = str(workspace_path)

        # Suppress deprecation warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Run lightweight telemetry
        try:
            disable_logging()
        except Exception as e:
            logger.error(f"Telemetry error: {e}")

    except Exception as e:
        logger.error(f"Bootup error: {str(e)}")
        raise
