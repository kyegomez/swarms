"""
Workspace directory naming for swarm autosave.

Writing into that directory is :class:`swarms.utils.workspace_manager.WorkspaceManager`,
which owns config, state, metadata and conversation output for every swarm.
"""

import os
import uuid
from datetime import datetime

from loguru import logger

from swarms.utils.workspace_utils import get_workspace_dir


def get_swarm_workspace_dir(
    class_name: str, swarm_name: str, use_timestamp: bool = True
) -> str:
    """
    Get the workspace directory path for a swarm instance.

    Creates a directory structure: workspace_dir/swarms/{class-name}/{swarm-name}-{timestamp or uuid}/

    Args:
        class_name (str): The name of the swarm class (e.g., "SwarmRouter", "GroupChat").
        swarm_name (str): The name of the swarm instance.
        use_timestamp (bool, optional): If True, use timestamp; if False, use UUID. Defaults to True.

    Returns:
        str: The full path to the swarm's workspace directory.
    """
    try:
        workspace_dir = get_workspace_dir()
    except ValueError:
        logger.warning(
            "WORKSPACE_DIR not set, cannot create swarm workspace directory"
        )
        return None

    # Sanitize names for filesystem compatibility
    class_name = _sanitize_name(class_name)
    swarm_name = _sanitize_name(swarm_name)

    # Create identifier (timestamp or UUID)
    if use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        identifier = timestamp
    else:
        identifier = uuid.uuid4().hex[:12]

    # Create directory structure: workspace_dir/swarms/{class-name}/{swarm-name}-{identifier}/
    swarm_dir = os.path.join(
        workspace_dir,
        "swarms",
        class_name,
        f"{swarm_name}-{identifier}",
    )

    # Create directory if it doesn't exist
    os.makedirs(swarm_dir, exist_ok=True)

    return swarm_dir


def _sanitize_name(name: str) -> str:
    """
    Sanitize a name for filesystem compatibility.

    Args:
        name (str): The name to sanitize.

    Returns:
        str: The sanitized name.
    """
    if not name:
        return "unnamed"
    # Replace invalid filesystem characters
    invalid_chars = '<>:"/\\|?*'
    sanitized = name
    for char in invalid_chars:
        sanitized = sanitized.replace(char, "_")
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(". ")
    # Replace spaces with hyphens
    sanitized = sanitized.replace(" ", "-")
    # Limit length
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    return sanitized
