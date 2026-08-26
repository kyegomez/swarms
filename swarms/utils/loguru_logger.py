import sys
from loguru import logger
from dotenv import load_dotenv

import os

load_dotenv()

# Ensure handlers are only configured once to prevent conflicts between modules.
_CONFIGURED = False

# Directory that current log handlers point to; may change during early boot.
_CONFIGURED_DIR = None

# Ceiling for a single per-module log file before it is rolled to ".1".
MODULE_LOG_MAX_BYTES = 10 * 1024 * 1024

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def get_log_dir() -> str:
    """
    Directory every log file is written to: ``{WORKSPACE_DIR}/logs``.

    Falls back to ``agent_workspace`` when ``WORKSPACE_DIR`` is unset, matching
    the default used elsewhere in the package.

    Returns:
        str: Path to the log directory.
    """
    workspace = os.getenv("WORKSPACE_DIR") or "agent_workspace"
    return os.path.join(workspace, "logs")


def _module_log_router(message) -> None:
    """
    Append each record to a file named after the module that emitted it.

    ``{workspace}/logs/graph_workflow.log`` then holds that module's lines and
    nothing else, so one component can be read in isolation while the combined
    log still shows the whole call chain interleaved.

    Routing on ``record["name"]`` rather than registering a handler per caller
    of :func:`initialize_logger` is what makes this complete: most modules take
    ``from loguru import logger`` directly and never call it, so a per-caller
    scheme would silently miss them — ``agent`` and ``conversation`` included.

    Files are opened on demand, so only modules that actually log get one.

    Args:
        message: The loguru message; ``message.record`` carries the metadata.
    """
    name = message.record["name"] or ""
    if not name.startswith("swarms"):
        return

    path = os.path.join(
        get_log_dir(), f"{name.rsplit('.', 1)[-1]}.log"
    )
    try:
        # Rotate by size. The combined log gets loguru's own time-based
        # rotation; these per-module views only need a ceiling so a chatty
        # module cannot fill the disk.
        if os.path.getsize(path) > MODULE_LOG_MAX_BYTES:
            os.replace(path, f"{path}.1")
    except OSError:
        pass

    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(message)
    except OSError:
        # Logging must never take down the caller.
        pass


def initialize_logger(log_folder: str = "swarms"):
    """
    Return the shared logger, configuring its handlers on first call.

    Args:
        log_folder (str): Name of the calling module. Accepted for backwards
            compatibility, but no longer used as a path: treating it as one
            created a directory per module in the working directory. The module
            is already recorded in each log line by ``{name}``.

    Returns:
        logger: The logger instance.
    """
    global _CONFIGURED, _CONFIGURED_DIR

    log_dir = get_log_dir()
    if _CONFIGURED and log_dir == _CONFIGURED_DIR:
        return logger
    try:
        os.makedirs(log_dir, exist_ok=True)
        file_logging = True
    except OSError as e:
        # WORKSPACE_DIR is caller-supplied and may be unwritable, or may
        # already exist as a file. Console logging still works, and being
        # unable to write logs must never stop `import swarms`.
        file_logging = False
        log_dir_error = e

    # Reset loguru handlers
    logger.remove()

    # Add console logging
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )

    if file_logging:
        # Add file logging (rotating)
        log_file_path = os.path.join(
            log_dir, "swarms_{time:YYYY-MM-DD}.log"
        )
        logger.add(
            log_file_path,
            rotation="1 day",
            retention="10 days",
            level="INFO",
            backtrace=True,
            diagnose=True,
            enqueue=True,
            format=LOG_FORMAT,
        )

        # Per-module files, routed by the emitting module, not the caller.
        logger.add(
            _module_log_router,
            level="INFO",
            format=LOG_FORMAT,
        )
    else:
        logger.warning(
            f"Log directory {log_dir!r} is unavailable "
            f"({log_dir_error}); logging to the console only."
        )

    _CONFIGURED = True
    _CONFIGURED_DIR = log_dir
    return logger
