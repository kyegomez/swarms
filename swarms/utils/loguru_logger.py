from loguru import logger

logger.add(
    "swarms.log",
    level="INFO",
    colorize=True,
    format="<green>{time}</green> <level>{message}</level>",
    backtrace=True,
    diagnose=True,
)


def loguru_logger(file_path: str = "swarms.log"):
    return logger.add(
        file_path,
        level="INFO",
        colorize=True,
        format="<green>{time}</green> <level>{message}</level>",
        backtrace=True,
        diagnose=True,
    )
