import logging
import functools


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
            logger.debug(
                f"Function {func.__name__} returned {result}"
            )
            return result
        except Exception as e:
            logger.error(
                f"Function {func.__name__} raised an exception: {e}"
            )
            raise

    return wrapper
