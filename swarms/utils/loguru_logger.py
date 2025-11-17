import sys
from loguru import logger


def initialize_logger(log_folder: str = "logs"):
    # Remove default handler and add a combined handler
    logger.remove()

    # Add a combined console and file handler
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        backtrace=True,
        diagnose=True,
        enqueue=True,
        # retention="10 days",  # Removed this line
    )

    return logger


# logger = initialize_logger()

# logger.info("Hello, world!")
