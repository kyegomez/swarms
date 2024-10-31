import os
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor

from swarms.telemetry.auto_upgrade_swarms import auto_update
from swarms.utils.disable_logging import disable_logging


def bootup():
    """Bootup swarms"""
    logging.disable(logging.CRITICAL)
    os.environ["WANDB_SILENT"] = "true"

    # Auto set workspace directory
    workspace_dir = os.path.join(os.getcwd(), "agent_workspace")
    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir)
    os.environ["WORKSPACE_DIR"] = workspace_dir

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Use ThreadPoolExecutor to run disable_logging and auto_update concurrently
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(disable_logging)
        executor.submit(auto_update)
