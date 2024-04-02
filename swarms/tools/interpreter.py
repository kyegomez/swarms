
import io
import sys

from swarms.utils.loguru_logger import logger


def execute_command(code):
    """
    Executes Python code and returns the output.

    Args:
        code (str): The Python code to execute.

    Returns:
        str: The output of the code.
    """
    # Create a string buffer to capture stdout and stderr
    buffer = io.StringIO()

    # Redirect stdout and stderr to the buffer
    sys.stdout = buffer
    sys.stderr = buffer

    try:
        # Execute the code
        exec(code)
    except Exception as e:
        # Log the error
        logger.error(f"Error executing code: {code}\n{str(e)}")
        return str(e)
    finally:
        # Restore stdout and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    # Get the output from the buffer
    output = buffer.getvalue()

    # Return the output
    return output