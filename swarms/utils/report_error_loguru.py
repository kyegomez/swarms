import datetime
import os
import platform
import sys
import traceback

from loguru import logger

# Configuring loguru to log to both the console and a file
logger.remove()  # Remove default logger configuration
logger.add(
    sys.stderr,
    level="ERROR",
    format="<red>{time}</red> - <level>{level}</level> - <level>{message}</level>",
)
logger.add(
    "error.log", level="ERROR", format="{time} - {level} - {message}"
)


def report_error(error: Exception) -> None:
    """
    Logs an error message and provides instructions for reporting the issue on Swarms GitHub
    or joining the community on Discord for real-time support.

    Args:
        error (Exception): The exception that occurred.

    Returns:
        None

    Raises:
        None
    """
    # Gather extensive context information
    context_info = {
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "stack_trace": traceback.format_exc(),
        "timestamp": datetime.datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "user": os.getenv("USER") or os.getenv("USERNAME"),
        "current_working_directory": os.getcwd(),
    }

    error_message = (
        f"\n"
        f"#########################################\n"
        f"#                                       #\n"
        f"#           ERROR DETECTED!             #\n"
        f"#                                       #\n"
        f"#                                       #\n"
        f"#                                       #\n"
        f"#            {error}                    #\n"
        f"#########################################\n"
        f"\n"
        f"Error Message: {context_info['exception_message']} ({context_info['exception_type']})\n"
        f"\n"
        f"Stack Trace:\n{context_info['stack_trace']}\n"
        f"\n"
        f"Context Information:\n"
        f"-----------------------------------------\n"
        f"Timestamp: {context_info['timestamp']}\n"
        f"Python Version: {context_info['python_version']}\n"
        f"Platform: {context_info['platform']}\n"
        f"Machine: {context_info['machine']}\n"
        f"Processor: {context_info['processor']}\n"
        f"User: {context_info['user']}\n"
        f"Current Working Directory: {context_info['current_working_directory']}\n"
        f"-----------------------------------------\n"
        f"\n"
        "Support"
        f"\n"
        f"\n"
        f"To report this issue, please visit the Swarms GitHub Issues page:\n"
        f"https://github.com/kyegomez/swarms/issues\n"
        f"\n"
        f"You can also join the Swarms community on Discord for real-time support:\n"
        f"https://discord.com/servers/agora-999382051935506503\n"
        f"\n"
        f"#########################################\n"
        f"-----------------------------------------\n"
    )

    logger.error(error_message)

    return None


# # Example usage:
# try:
#     # Simulate an error
#     raise ValueError("An example error")
# except Exception as e:
#     report_error(e)
