import os
import uuid
import sys
from loguru import logger


def initialize_logger(log_folder: str = "logs"):
    AGENT_WORKSPACE = "agent_workspace"

    # Check if WORKSPACE_DIR is set, if not, set it to AGENT_WORKSPACE
    if "WORKSPACE_DIR" not in os.environ:
        os.environ["WORKSPACE_DIR"] = AGENT_WORKSPACE

    # Create a folder within the agent_workspace
    log_folder_path = os.path.join(
        os.getenv("WORKSPACE_DIR"), log_folder
    )
    if not os.path.exists(log_folder_path):
        os.makedirs(log_folder_path)

    # Generate a unique identifier for the log file
    uuid_for_log = str(uuid.uuid4())
    log_file_path = os.path.join(
        log_folder_path, f"{log_folder}_{uuid_for_log}.log"
    )

    # Remove default handler and add custom handlers
    logger.remove()

    # Add console handler with colors
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    # Add file handler
    logger.add(
        log_file_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
        backtrace=True,
        diagnose=True,
        enqueue=True,
        retention="10 days",
        # compression="zip",
    )

    return logger
