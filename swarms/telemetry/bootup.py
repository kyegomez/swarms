import os
import warnings
from pathlib import Path
from swarms.utils.disable_logging import disable_logging
from loguru import logger


def _prepare_workspace() -> None:
    """
    Resolve ``WORKSPACE_DIR`` and make sure that directory exists.

    Defers to :func:`ensure_workspace_env` rather than defaulting the variable
    here as well, so bootup and ``WorkspaceManager`` cannot drift apart.

    A caller-supplied path that cannot be created falls back to the default
    instead of raising: an unusable workspace should not stop ``import
    swarms``, which is what an uncaught ``OSError`` here would do.
    """
    # Imported here, not at module scope, which would run initialize_logger before WORKSPACE_DIR settles.
    from swarms.utils.workspace_manager import ensure_workspace_env

    fallback = Path.cwd() / "agent_workspace"
    workspace = ensure_workspace_env() or str(fallback)

    try:
        Path(workspace).mkdir(parents=True, exist_ok=True)
        return
    except OSError as e:
        # Unwritable paths, or WORKSPACE_DIR existing as a file
        logger.warning(
            f"WORKSPACE_DIR={workspace!r} could not be created ({e}); "
            f"falling back to {fallback}"
        )

    os.environ["WORKSPACE_DIR"] = str(fallback)
    try:
        fallback.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning(
            f"Default workspace {fallback} is unavailable too: {e}"
        )


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
        _prepare_workspace()

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
