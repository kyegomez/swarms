import sys
from loguru import logger
from dotenv import load_dotenv

import os

load_dotenv()


def initialize_logger(log_folder: str = None):
    """
    Initialize the logger for the application.

    Args:
        log_folder (str): The folder to save the logs to. Defaults to
            ``WORKSPACE_DIR``, or ``"logs"`` when that is unset.

    Returns:
        logger: The logger instance.
    """
    # Set log folder, fallback to defaults
    log_folder = log_folder or os.getenv("WORKSPACE_DIR") or "logs"

    # Create log folder if it doesn't exist
    if not os.path.exists(log_folder):
        os.makedirs(log_folder, exist_ok=True)

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

    # Add file logging (rotating)
    log_file_path = os.path.join(
        log_folder, "log_{time:YYYY-MM-DD}.log"
    )
    logger.add(
        log_file_path,
        rotation="1 day",
        retention="10 days",
        level="INFO",
        backtrace=True,
        diagnose=True,
        enqueue=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    return logger
