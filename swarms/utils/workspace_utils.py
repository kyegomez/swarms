import os
from functools import lru_cache
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


@lru_cache(maxsize=8)
def _resolve_workspace_dir(workspace_dir):
    """
    Report an unset workspace directory once per value.

    Args:
        workspace_dir (str): The value of the WORKSPACE_DIR environment variable.

    Returns:
        str: The value it was given.
    """
    if not workspace_dir:
        logger.error(
            "WORKSPACE_DIR environment variable is not set. Please set WORKSPACE_DIR in your environment or in a .env file. Example: WORKSPACE_DIR='agent_workspace'"
        )
    return workspace_dir


def get_workspace_dir():
    """
    Retrieve the workspace directory path from the WORKSPACE_DIR environment variable.

    The variable is read on every call, so a WORKSPACE_DIR set after swarms
    was imported is honoured. The cache only keeps the warning about an
    unset variable from repeating.

    Returns:
        str: The absolute path of the workspace directory.
    """
    return _resolve_workspace_dir(os.getenv("WORKSPACE_DIR"))


get_workspace_dir.cache_clear = _resolve_workspace_dir.cache_clear
