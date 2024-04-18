import datetime
import functools
import logging

logger = logging.getLogger()
formatter = logging.Formatter("%(message)s")

ch = logging.StreamHandler()
ch.setFormatter(formatter)

logger.addHandler(ch)
logger.setLevel(logging.DEBUG)


def log_wrapper(func):
    """
    A decorator that logs the inputs, outputs, and any exceptions of the function it wraps.

    Args:
        func (callable): The function to wrap.

    Returns:
        callable: The wrapped function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(
            f"Calling function {func.__name__} with args {args} and"
            f" kwargs {kwargs}"
        )
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Function {func.__name__} returned {result}")
            return result
        except Exception as e:
            logger.error(
                f"Function {func.__name__} raised an exception: {e}"
            )
            raise

    return wrapper


class Logger:
    """
    A utility class for logging messages with timestamps and levels.

    Attributes:
        logger (logging.Logger): The logger object used for logging messages.
        formatter (logging.Formatter): The formatter object used to format log messages.
        ch (logging.StreamHandler): The stream handler object used to handle log messages.
    """

    logger = logging.getLogger(__name__)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(message)s"
    )
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    @staticmethod
    def log(level, task, message):
        """
        Logs a message with the specified level, task, and message.

        Args:
            level (int): The logging level of the message.
            task (str): The task associated with the message.
            message (str): The message to be logged.
        """
        timestamp = datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")
        formatted_message = (
            f"[{timestamp}] {level:<8} {task}\n{' ' * 29}{message}"
        )
        Logger.logger.log(level, formatted_message)
