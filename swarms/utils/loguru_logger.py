from loguru import logger

logger.add(
    "MessagePool.log",
    level="INFO",
    colorize=True,
    format="<green>{time}</green> <level>{message}</level>",
    backtrace=True,
    diagnose=True,
)
