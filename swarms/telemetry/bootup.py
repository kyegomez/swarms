import os
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from swarms.utils.disable_logging import disable_logging


def bootup():
    """Bootup swarms"""
    try:
        logging.disable(logging.CRITICAL)
        os.environ["WANDB_SILENT"] = "true"

        # Auto set workspace directory
        workspace_dir = os.path.join(os.getcwd(), "agent_workspace")
        if not os.path.exists(workspace_dir):
            os.makedirs(workspace_dir, exist_ok=True)
        os.environ["WORKSPACE_DIR"] = workspace_dir

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Use ThreadPoolExecutor to run disable_logging and auto_update concurrently
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(disable_logging)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
