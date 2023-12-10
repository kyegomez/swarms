import subprocess
import sys
import traceback
import functools

try:
    import phoenix as px
except Exception as error:
    print(f"Error importing phoenix: {error}")
    print("Please install phoenix: pip install phoenix")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "arize-mlflow"]
    )


def phoenix_trace_decorator(doc_string):
    """Phoenix trace decorator.


    Args:
        doc_string (_type_): doc string for the function


    Example:
        >>> @phoenix_trace_decorator(
        >>>     "This is a doc string"
        >>> )
        >>> def test_function():
        >>>     print("Hello world")
        >>>
        >>> test_function()


    # Example of using the decorator
    @phoenix_trace_decorator("This function does XYZ and is traced by Phoenix.")
    def my_function(param1, param2):
        # Function implementation
        pass
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Start phoenix session for tracing
            session = px.active_session() or px.launch_app()

            try:
                # Attempt to execute the function
                result = func(*args, **kwargs)
                return result
            except Exception as error:
                error_info = traceback.format_exc()
                session.trace_exception(
                    exception=error, error_info=error_info
                )
                raise

        # Atteach docs to wrapper func
        wrapper.__doc__ = doc_string
        return wrapper

    return decorator
