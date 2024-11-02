import os
from loguru import logger


WORKSPACE_DIR = os.getenv("WORKSPACE_DIR")

logger.add(
    os.path.join(WORKSPACE_DIR, "swarms.log"),
    level="INFO",
    colorize=True,
    backtrace=True,
    diagnose=True,
)


def loguru_logger(file_path: str = "swarms.log"):
    return logger.add(
        os.path.join(WORKSPACE_DIR, file_path),
        level="INFO",
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
