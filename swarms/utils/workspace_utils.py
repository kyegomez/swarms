import os
from loguru import logger


def check_if_workspace_dir_exists():
    """
    Check if the workspace directory exists.

    Returns:
        bool: True if the workspace directory exists, False otherwise.

    Raises:
        ValueError: If the workspace directory environment variable is not set.
    """
    workspace_dir = get_workspace_dir()
    return os.path.exists(workspace_dir)


def get_workspace_dir():
    """
    Retrieve the workspace directory path from the WORKSPACE_DIR environment variable.

    Read on every call. This used to be lru_cached with no arguments, so the
    first resolved value was frozen for the life of the process and later
    changes to WORKSPACE_DIR had no effect. os.getenv is a dict lookup, so
    there was nothing to cache.

    Returns:
        str: The absolute path of the workspace directory.

    Raises:
        ValueError: If the WORKSPACE_DIR environment variable is not set.
    """
    workspace_dir = os.getenv("WORKSPACE_DIR")
    if not workspace_dir:
        logger.error(
            "WORKSPACE_DIR environment variable is not set. Please set WORKSPACE_DIR in your environment or in a .env file. Example: WORKSPACE_DIR='agent_workspace'"
        )
    return workspace_dir
