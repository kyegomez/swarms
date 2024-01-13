def try_except_wrapper(func):
    """
    A decorator that wraps a function with a try-except block.
    It catches any exception that occurs during the execution of the function,
    prints an error message, and returns None.
    It also prints a message indicating the exit of the function.

    Args:
        func (function): The function to be wrapped.

    Returns:
        function: The wrapped function.
    """

    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as error:
            print(
                f"An error occurred in function {func.__name__}:"
                f" {error}"
            )
            return None
        finally:
            print(f"Exiting function: {func.__name__}")

    return wrapper
