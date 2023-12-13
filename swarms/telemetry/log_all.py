import logging
import types

# Set up logging
logging.basicConfig(level=logging.INFO)


# Log all calls to functions in this module
def log_all_calls(module):
    """
    Decorate all functions of a module to log calls to them.
    """
    for name, obj in vars(module).items():
        if isinstance(obj, types.FunctionType):
            setattr(module, name, log_calls(obj))


# Log all calls to a function
def log_calls(func):
    """
    Decorate a function to log calls to it.
    """

    def wrapper(*args, **kwargs):
        logging.info(
            f"Calling function {func.__name__} with args {args} and"
            f" kwargs {kwargs}"
        )
        result = func(*args, **kwargs)
        logging.info(f"Function {func.__name__} returned {result}")
        return result

    return wrapper
