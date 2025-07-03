import time
from typing import Any, Callable, Type, Union, Tuple
from loguru import logger


def retry_function(
    func: Callable,
    *args: Any,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Union[
        Type[Exception], Tuple[Type[Exception], ...]
    ] = Exception,
    **kwargs: Any,
) -> Any:
    """
    A function that retries another function if it raises specified exceptions.

    Args:
        func (Callable): The function to retry
        *args: Positional arguments to pass to the function
        max_retries (int): Maximum number of retries before giving up. Defaults to 3.
        delay (float): Initial delay between retries in seconds. Defaults to 1.0.
        backoff_factor (float): Multiplier applied to delay between retries. Defaults to 2.0.
        exceptions (Exception or tuple): Exception(s) that trigger a retry. Defaults to Exception.
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Any: The return value of the function if successful

    Example:
        def fetch_data(url: str) -> dict:
            return requests.get(url).json()

        # Retry the fetch_data function
        result = retry_function(
            fetch_data,
            "https://api.example.com",
            max_retries=3,
            exceptions=(ConnectionError, TimeoutError)
        )
    """
    retries = 0
    current_delay = delay

    while True:
        try:
            return func(*args, **kwargs)
        except exceptions as e:
            retries += 1
            if retries > max_retries:
                logger.error(
                    f"Function {func.__name__} failed after {max_retries} retries. "
                    f"Final error: {str(e)}"
                )
                raise

            logger.warning(
                f"Retry {retries}/{max_retries} for function {func.__name__} "
                f"after error: {str(e)}. "
                f"Waiting {current_delay} seconds..."
            )

            time.sleep(current_delay)
            current_delay *= backoff_factor
