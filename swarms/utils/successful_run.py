from loguru import logger
import sys
import platform
import os
import datetime

# Configuring loguru to log to both the console and a file
logger.remove()  # Remove default logger configuration
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time}</green> - <level>{level}</level> - <level>{message}</level>",
)

logger.add(
    "info.log", level="INFO", format="{time} - {level} - {message}"
)


def log_success_message() -> None:
    """
    Logs a success message with instructions for sharing agents on the Swarms Agent Explorer and joining the community for assistance.

    Returns:
        None

    Raises:
        None
    """
    # Gather extensive context information
    context_info = {
        "timestamp": datetime.datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "user": os.getenv("USER") or os.getenv("USERNAME"),
        "current_working_directory": os.getcwd(),
    }

    success_message = (
        f"\n"
        f"#########################################\n"
        f"#                                       #\n"
        f"#        SUCCESSFUL RUN DETECTED!       #\n"
        f"#                                       #\n"
        f"#########################################\n"
        f"\n"
        f"Your task completed successfully!\n"
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
        f"Share your agents on the Swarms Agent Explorer with friends:\n"
        f"https://swarms.world/platform/explorer\n"
        f"\n"
        f"Join the Swarms community if you want assistance or help debugging:\n"
        f"https://discord.gg/uzu63HQx\n"
        f"\n"
        f"#########################################\n"
    )

    logger.info(success_message)


# Example usage:
# log_success_message()
