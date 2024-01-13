import time
from functools import wraps
from typing import Callable


def metrics_decorator(func: Callable):
    """Metrics decorator for LLM

    Args:
        func (Callable): The function to decorate

    Example:
    >>> @metrics_decorator
    >>> def my_function():
    >>>     return "Hello, world!"
    >>> my_function()

    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Time to First Token
        start_time = time.time()
        result = func(self, *args, **kwargs)
        first_token_time = time.time()

        # Generation Latency
        end_time = time.time()

        # Throughput (assuming the function returns a list of tokens)
        throughput = len(result) / (end_time - start_time)

        return f"""
        Time to First Token: {first_token_time - start_time}
        Generation Latency: {end_time - start_time}
        Throughput: {throughput}
        """

    return wrapper
