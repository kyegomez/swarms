from swarms.utils.loguru_logger import logger


def try_except_wrapper(func, verbose: bool = False):
    """
    A decorator that wraps a function with a try-except block.
    It catches any exception that occurs during the execution of the function,
    prints an error message, and returns None.
    It also prints a message indicating the exit of the function.

    Args:
        func (function): The function to be wrapped.

    Returns:
        function: The wrapped function.

    Examples:
    >>> @try_except_wrapper(verbose=True)
    ... def divide(a, b):
    ...     return a / b
    >>> divide(1, 0)
    An error occurred in function divide: division by zero
    Exiting function: divide
    """

    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as error:
            if verbose:
                logger.error(
                    f"An error occurred in function {func.__name__}:"
                    f" {error}"
                )
            else:
                print(
                    f"An error occurred in function {func.__name__}:"
                    f" {error}"
                )
                return None
        finally:
            print(f"Exiting function: {func.__name__}")

    return wrapper
