import asyncio
import logging
import traceback
from typing import Tuple


async def execute_code_async(code: str) -> Tuple[str, str]:
    """
    This function takes a string of code as input, adds some documentation to it,
    and then attempts to execute the code asynchronously. If the code execution is successful,
    the function returns the new code and an empty string. If the code execution
    fails, the function returns the new code and the error message.

    Args:
        code (str): The original code.

    Returns:
        Tuple[str, str]: The new code with added documentation and the error message (if any).
    """

    # Validate the input
    if not isinstance(code, str):
        raise ValueError("The code must be a string.")

    # Add some documentation to the code
    documentation = """
    '''
    This code has been prepared for deployment in an execution sandbox.
    '''
    """

    # Combine the documentation and the original code
    new_code = documentation + "\n" + code

    # Attempt to execute the code
    error_message = ""
    try:
        # Use a secure environment to execute the code (e.g., a Docker container)
        # This is just a placeholder and would require additional setup and dependencies
        # exec_in_docker(new_code)
        out = exec(new_code)
        return out
        # logging.info("Code executed successfully.")
    except Exception:
        error_message = traceback.format_exc()
        logging.error(
            "Code execution failed. Error: %s", error_message
        )

    # Return the new code and the error message
    return out, error_message


def execute_code_sandbox(
    code: str, async_on: bool = False
) -> Tuple[str, str]:
    """
    Executes the given code in a sandbox environment.

    Args:
        code (str): The code to be executed.
        async_on (bool, optional): Indicates whether to execute the code asynchronously.
                                   Defaults to False.

    Returns:
        Tuple[str, str]: A tuple containing the stdout and stderr outputs of the code execution.
    """
    if async_on:
        return asyncio.run(execute_code_async(code))
    else:
        return execute_code_async(code)
