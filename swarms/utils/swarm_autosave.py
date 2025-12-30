"""
Utility functions for autosaving swarm configurations and state.

This module provides functions to automatically save swarm configurations,
state, and metadata to organized directory structures within the workspace.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger

from swarms.utils.file_processing import create_file_in_folder
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


def save_swarm_config(
    swarm_instance: Any,
    swarm_workspace_dir: str,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Save swarm configuration to config.json.

    Args:
        swarm_instance (Any): The swarm instance to save configuration for.
        swarm_workspace_dir (str): The workspace directory for this swarm.
        additional_metadata (Optional[Dict[str, Any]]): Additional metadata to include.

    Returns:
        Optional[str]: Path to the saved config file, or None if failed.
    """
    if not swarm_workspace_dir:
        return None

    try:
        # Get configuration dictionary
        if hasattr(swarm_instance, "to_dict"):
            config_dict = swarm_instance.to_dict()
        else:
            # Fallback: create dict from instance attributes
            config_dict = {
                k: v
                for k, v in swarm_instance.__dict__.items()
                if not k.startswith("_") and not callable(v)
            }

        # Add metadata
        config_dict["_autosave_metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "class_name": swarm_instance.__class__.__name__,
            "swarm_name": getattr(swarm_instance, "name", "unnamed"),
            "swarm_id": getattr(swarm_instance, "id", None),
        }

        # Add additional metadata if provided
        if additional_metadata:
            config_dict["_autosave_metadata"].update(
                additional_metadata
            )

        # Convert to JSON string
        config_json = json.dumps(config_dict, indent=2, default=str)

        # Save to file
        config_path = create_file_in_folder(
            swarm_workspace_dir, "config.json", config_json
        )

        if config_path:
            logger.debug(f"Saved swarm config to {config_path}")
        return config_path

    except Exception as e:
        logger.warning(f"Failed to save swarm config: {e}")
        return None


def save_swarm_state(
    swarm_instance: Any,
    swarm_workspace_dir: str,
    state_data: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Save swarm state to state.json.

    Args:
        swarm_instance (Any): The swarm instance to save state for.
        swarm_workspace_dir (str): The workspace directory for this swarm.
        state_data (Optional[Dict[str, Any]]): Additional state data to include.

    Returns:
        Optional[str]: Path to the saved state file, or None if failed.
    """
    if not swarm_workspace_dir:
        return None

    try:
        # Build state dictionary
        state_dict = {
            "timestamp": datetime.now().isoformat(),
            "swarm_name": getattr(swarm_instance, "name", "unnamed"),
            "swarm_id": getattr(swarm_instance, "id", None),
            "swarm_type": getattr(swarm_instance, "swarm_type", None),
        }

        # Add conversation history if available
        if hasattr(swarm_instance, "swarm") and hasattr(
            swarm_instance.swarm, "conversation"
        ):
            try:
                conversation = swarm_instance.swarm.conversation
                if hasattr(conversation, "to_dict"):
                    state_dict["conversation"] = (
                        conversation.to_dict()
                    )
                elif hasattr(conversation, "conversation_history"):
                    state_dict["conversation"] = (
                        conversation.conversation_history
                    )
            except Exception as e:
                logger.debug(f"Could not save conversation: {e}")

        # Add logs if available
        if hasattr(swarm_instance, "logs"):
            state_dict["logs"] = [
                log if not callable(log) else str(log)
                for log in swarm_instance.logs
            ]

        # Add additional state data if provided
        if state_data:
            state_dict.update(state_data)

        # Convert to JSON string
        state_json = json.dumps(state_dict, indent=2, default=str)

        # Save to file
        state_path = create_file_in_folder(
            swarm_workspace_dir, "state.json", state_json
        )

        if state_path:
            logger.debug(f"Saved swarm state to {state_path}")
        return state_path

    except Exception as e:
        logger.warning(f"Failed to save swarm state: {e}")
        return None


def save_swarm_metadata(
    swarm_instance: Any,
    swarm_workspace_dir: str,
    execution_result: Optional[Any] = None,
    execution_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Save swarm execution metadata to metadata.json.

    Args:
        swarm_instance (Any): The swarm instance to save metadata for.
        swarm_workspace_dir (str): The workspace directory for this swarm.
        execution_result (Optional[Any]): The result of the execution.
        execution_metadata (Optional[Dict[str, Any]]): Additional execution metadata.

    Returns:
        Optional[str]: Path to the saved metadata file, or None if failed.
    """
    if not swarm_workspace_dir:
        return None

    try:
        metadata_dict = {
            "execution_timestamp": datetime.now().isoformat(),
            "swarm_name": getattr(swarm_instance, "name", "unnamed"),
            "swarm_id": getattr(swarm_instance, "id", None),
            "swarm_type": getattr(swarm_instance, "swarm_type", None),
            "max_loops": getattr(swarm_instance, "max_loops", None),
            "agents_count": (
                len(swarm_instance.agents)
                if hasattr(swarm_instance, "agents")
                else 0
            ),
        }

        # Add execution result summary
        if execution_result is not None:
            if isinstance(execution_result, (str, int, float, bool)):
                metadata_dict["execution_result_summary"] = str(
                    execution_result
                )
            elif isinstance(execution_result, (list, dict)):
                metadata_dict["execution_result_summary"] = {
                    "type": type(execution_result).__name__,
                    "length": len(execution_result),
                }
            else:
                metadata_dict["execution_result_summary"] = str(
                    type(execution_result).__name__
                )

        # Add additional execution metadata if provided
        if execution_metadata:
            metadata_dict.update(execution_metadata)

        # Convert to JSON string
        metadata_json = json.dumps(
            metadata_dict, indent=2, default=str
        )

        # Save to file
        metadata_path = create_file_in_folder(
            swarm_workspace_dir, "metadata.json", metadata_json
        )

        if metadata_path:
            logger.debug(f"Saved swarm metadata to {metadata_path}")
        return metadata_path

    except Exception as e:
        logger.warning(f"Failed to save swarm metadata: {e}")
        return None


def autosave_swarm(
    swarm_instance: Any,
    swarm_workspace_dir: Optional[str] = None,
    save_config: bool = True,
    save_state: bool = True,
    save_metadata: bool = False,
    execution_result: Optional[Any] = None,
    additional_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[str]]:
    """
    Automatically save swarm configuration, state, and metadata.

    Args:
        swarm_instance (Any): The swarm instance to save.
        swarm_workspace_dir (Optional[str]): The workspace directory. If None, will be generated.
        save_config (bool): Whether to save configuration. Defaults to True.
        save_state (bool): Whether to save state. Defaults to True.
        save_metadata (bool): Whether to save metadata. Defaults to False.
        execution_result (Optional[Any]): The result of execution for metadata.
        additional_data (Optional[Dict[str, Any]]): Additional data to include.

    Returns:
        Dict[str, Optional[str]]: Dictionary with paths to saved files.
    """
    results = {"config": None, "state": None, "metadata": None}

    # Generate workspace directory if not provided
    if swarm_workspace_dir is None:
        class_name = swarm_instance.__class__.__name__
        swarm_name = getattr(swarm_instance, "name", "unnamed")
        swarm_workspace_dir = get_swarm_workspace_dir(
            class_name, swarm_name, use_timestamp=True
        )

    if not swarm_workspace_dir:
        logger.warning("Could not create swarm workspace directory")
        return results

    # Save configuration
    if save_config:
        results["config"] = save_swarm_config(
            swarm_instance, swarm_workspace_dir, additional_data
        )

    # Save state
    if save_state:
        state_data = (
            additional_data.get("state", {})
            if additional_data
            else {}
        )
        results["state"] = save_swarm_state(
            swarm_instance, swarm_workspace_dir, state_data
        )

    # Save metadata
    if save_metadata:
        execution_metadata = (
            additional_data.get("execution_metadata", {})
            if additional_data
            else {}
        )
        results["metadata"] = save_swarm_metadata(
            swarm_instance,
            swarm_workspace_dir,
            execution_result,
            execution_metadata,
        )

    return results
