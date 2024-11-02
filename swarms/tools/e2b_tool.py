import subprocess
import sys
from loguru import logger
from typing import Tuple, Union, List
from e2b_code_interpreter import CodeInterpreter

# load_dotenv()


# Helper function to lazily install the package if not found
def lazy_install(package: str) -> None:
    try:
        __import__(package)
    except ImportError:
        logger.warning(f"{package} not found. Installing now...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package]
        )


# Ensure e2b_code_interpreter is installed lazily
lazy_install("e2b_code_interpreter")


def code_interpret(
    code_interpreter: CodeInterpreter, code: str
) -> Union[Tuple[List[str], List[str]], None]:
    """
    Runs AI-generated code using the provided CodeInterpreter and logs the process.

    Args:
        code_interpreter (CodeInterpreter): An instance of the CodeInterpreter class.
        code (str): The code string to be executed.

    Returns:
        Union[Tuple[List[str], List[str]], None]: A tuple of (results, logs) if successful,
        or None if an error occurred.

    Raises:
        ValueError: If the code or code_interpreter is invalid.
    """
    if not isinstance(code_interpreter, CodeInterpreter):
        logger.error("Invalid CodeInterpreter instance provided.")
        raise ValueError(
            "code_interpreter must be an instance of CodeInterpreter."
        )
    if not isinstance(code, str) or not code.strip():
        logger.error("Invalid code provided.")
        raise ValueError("code must be a non-empty string.")

    logger.info(
        f"\n{'='*50}\n> Running the following AI-generated code:\n{code}\n{'='*50}"
    )

    try:
        exec_result = code_interpreter.notebook.exec_cell(
            code,
            # on_stderr=lambda stderr: logger.error(f"[Code Interpreter stderr] {stderr}"),
            # on_stdout=lambda stdout: logger.info(f"[Code Interpreter stdout] {stdout}")
        )

        if exec_result.error:
            logger.error(
                f"[Code Interpreter error] {exec_result.error}"
            )
            return None
        else:
            logger.success("Code executed successfully.")
            # return exec_result.results, exec_result.logs
            # return exec_result.results
            prompt = f"{exec_result.results}: {exec_result.logs}"
            return prompt

    except Exception:
        logger.exception(
            "An error occurred during code interpretation."
        )
        return None


# # from e2b_code_interpreter import CodeInterpreter

# interpreter = CodeInterpreter()
# code = "print('Hello, World!')"

# result = code_interpret(interpreter, code)

# if result:
#     results = result
#     print("Execution Results:", results)
#     # print("Execution Logs:", logs)
