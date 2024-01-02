from typing import Callable, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def block(
    function: Callable[..., Any],
    device: str = None,
    verbose: bool = False,
) -> Callable[..., Any]:
    """
    A decorator that transforms a function into a block.

    Args:
        function (Callable[..., Any]): The function to transform.

    Returns:
        Callable[..., Any]: The transformed function.
    """

    def wrapper(*args, **kwargs):
        # Here you can add code to execute the function on various hardwares
        # For now, we'll just call the function normally
        try:
            return function(*args, **kwargs)
        except Exception as error:
            logger.error(f"Error in {function.__name__}: {error}")
            raise error

    # Set the wrapper function's name and docstring to those of the original function
    wrapper.__name__ = function.__name__
    wrapper.__doc__ = function.__doc__

    return wrapper
