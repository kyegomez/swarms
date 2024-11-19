import os
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor

from swarms.telemetry.auto_upgrade_swarms import auto_update
from swarms.utils.disable_logging import disable_logging
from swarms.utils.workspace_manager import WorkspaceManager


def bootup():
    """Bootup swarms"""
    try:
        logging.disable(logging.CRITICAL)
        os.environ["WANDB_SILENT"] = "true"

    # Set workspace directory using WorkspaceManager
    try:
        workspace_dir = WorkspaceManager.get_workspace_dir()
        os.environ["WORKSPACE_DIR"] = workspace_dir
    except Exception as e:
        print(f"Error setting up workspace directory: {e}")
        return

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Use ThreadPoolExecutor to run disable_logging and auto_update concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(disable_logging)
            executor.submit(auto_update)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
