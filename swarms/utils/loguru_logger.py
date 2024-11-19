import os
import uuid
from loguru import logger
from swarms.utils.workspace_manager import WorkspaceManager

try:
    WORKSPACE_DIR = WorkspaceManager.get_workspace_dir()
    
    logger.add(
        os.path.join(WORKSPACE_DIR, "swarms.log"),
        level="INFO",
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

except Exception as e:
    logger.error(f"Failed to initialize logger: {e}")

def loguru_logger(file_path: str = "swarms.log"):
    try:
        return logger.add(
            os.path.join(WORKSPACE_DIR, file_path),
            level="INFO",
            colorize=True,
            backtrace=True,
            diagnose=True,
        )
    except Exception as e:
        logger.error(f"Failed to create logger for {file_path}: {e}")
        raise
