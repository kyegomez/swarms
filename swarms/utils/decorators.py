import functools
import logging
import threading
import warnings


def retry_decorator(max_retries: int = 5):
    """
    Decorator that retries a function a specified number of times if an exception occurs.

    Args:
        max_retries (int): The maximum number of times to retry the function.

    Returns:
        function: The decorated function.

    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as error:
                    logging.error(
                        f" Error in {func.__name__}:"
                        f" {str(error)} Retrying ...."
                    )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def singleton_decorator(cls):
    instances = {}

    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


def synchronized_decorator(func):
    func.__lock__ = threading.Lock()

    def wrapper(*args, **kwargs):
        with func.__lock__:
            return func(*args, **kwargs)

    return wrapper


def deprecated_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{func.__name__} is deprecated",
            category=DeprecationWarning,
        )
        return func(*args, **kwargs)

    return wrapper


def validate_inputs_decorator(validator):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not validator(*args, **kwargs):
                raise ValueError("Invalid Inputs")
            return func(*args, **kwargs)

        return wrapper

    return decorator
