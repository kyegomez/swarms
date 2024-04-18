"""Logging modules"""

import json
import logging
import os
import random
import re
import time
from logging import LogRecord
from typing import Any

from colorama import Fore, Style

from swarms.utils.apa import Action, ToolCallStatus


# from autogpt.speech import say_text
class JsonFileHandler(logging.FileHandler):
    def __init__(self, filename, mode="a", encoding=None, delay=False):
        """
        Initializes a new instance of the class with the specified file name, mode, encoding, and delay settings.

        Parameters:
            filename (str): The name of the file to be opened.
            mode (str, optional): The mode in which the file is opened. Defaults to "a" (append).
            encoding (str, optional): The encoding used to read or write the file. Defaults to None.
            delay (bool, optional): If True, the file opening is delayed until the first IO operation. Defaults to False.

        Returns:
            None
        """
        super().__init__(filename, mode, encoding, delay)

    def emit(self, record):
        """
        Writes the formatted log record to a JSON file.

        Parameters:
            record (LogRecord): The log record to be emitted.

        Returns:
            None
        """
        json_data = json.loads(self.format(record))
        with open(self.baseFilename, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)


class JsonFormatter(logging.Formatter):
    def format(self, record):
        """
        Format the given record and return the message.

        Args:
            record (object): The log record to be formatted.

        Returns:
            str: The formatted message from the record.
        """
        return record.msg


class Logger:
    """
    Logger that handle titles in different colors.
    Outputs logs in console, activity.log, and errors.log
    For console handler: simulates typing
    """

    def __init__(self):
        """
        Initializes the class and sets up the logging configuration.

        Args:
            None

        Returns:
            None
        """
        # create log directory if it doesn't exist
        this_files_dir_path = os.path.dirname(__file__)
        log_dir = os.path.join(this_files_dir_path, "../logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = "activity.log"
        error_file = "error.log"

        console_formatter = AutoGptFormatter("%(title_color)s %(message)s")

        # Create a handler for console which simulate typing
        self.typing_console_handler = TypingConsoleHandler()
        # self.typing_console_handler = ConsoleHandler()
        self.typing_console_handler.setLevel(logging.INFO)
        self.typing_console_handler.setFormatter(console_formatter)

        # Create a handler for console without typing simulation
        self.console_handler = ConsoleHandler()
        self.console_handler.setLevel(logging.DEBUG)
        self.console_handler.setFormatter(console_formatter)

        # Info handler in activity.log
        self.file_handler = logging.FileHandler(
            os.path.join(log_dir, log_file), "a", "utf-8"
        )
        self.file_handler.setLevel(logging.DEBUG)
        info_formatter = AutoGptFormatter(
            "%(asctime)s %(levelname)s %(title)s %(message_no_color)s"
        )
        self.file_handler.setFormatter(info_formatter)

        # Error handler error.log
        error_handler = logging.FileHandler(
            os.path.join(log_dir, error_file), "a", "utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = AutoGptFormatter(
            "%(asctime)s %(levelname)s"
            " %(module)s:%(funcName)s:%(lineno)d %(title)s"
            " %(message_no_color)s"
        )
        error_handler.setFormatter(error_formatter)

        self.typing_logger = logging.getLogger("TYPER")
        self.typing_logger.addHandler(self.typing_console_handler)
        # self.typing_logger.addHandler(self.console_handler)
        self.typing_logger.addHandler(self.file_handler)
        self.typing_logger.addHandler(error_handler)
        self.typing_logger.setLevel(logging.DEBUG)

        self.logger = logging.getLogger("LOGGER")
        self.logger.addHandler(self.console_handler)
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(error_handler)
        self.logger.setLevel(logging.DEBUG)

        self.json_logger = logging.getLogger("JSON_LOGGER")
        self.json_logger.addHandler(self.file_handler)
        self.json_logger.addHandler(error_handler)
        self.json_logger.setLevel(logging.DEBUG)

        self.speak_mode = False
        self.chat_plugins = []

    def typewriter_log(
        self,
        title="",
        title_color="",
        content="",
        speak_text=False,
        level=logging.INFO,
    ):
        """
        Logs a message to the typewriter.

        Args:
            title (str, optional): The title of the log message. Defaults to "".
            title_color (str, optional): The color of the title. Defaults to "".
            content (str or list, optional): The content of the log message. Defaults to "".
            speak_text (bool, optional): Whether to speak the log message. Defaults to False.
            level (int, optional): The logging level of the message. Defaults to logging.INFO.

        Returns:
            None
        """
        for plugin in self.chat_plugins:
            plugin.report(f"{title}. {content}")

        if content:
            if isinstance(content, list):
                content = " ".join(content)
        else:
            content = ""

        self.typing_logger.log(
            level,
            content,
            extra={"title": title, "color": title_color},
        )

    def debug(
        self,
        message,
        title="",
        title_color="",
    ):
        """
        Logs a debug message.

        Args:
            message (str): The debug message to log.
            title (str, optional): The title of the log message. Defaults to "".
            title_color (str, optional): The color of the log message title. Defaults to "".

        Returns:
            None
        """
        self._log(title, title_color, message, logging.DEBUG)

    def info(
        self,
        message,
        title="",
        title_color="",
    ):
        """
        Logs an informational message.

        Args:
            message (str): The message to be logged.
            title (str, optional): The title of the log message. Defaults to "".
            title_color (str, optional): The color of the log title. Defaults to "".

        Returns:
            None
        """
        self._log(title, title_color, message, logging.INFO)

    def warn(
        self,
        message,
        title="",
        title_color="",
    ):
        """
        Logs a warning message.

        Args:
            message (str): The warning message.
            title (str, optional): The title of the warning message. Defaults to "".
            title_color (str, optional): The color of the title. Defaults to "".
        """
        self._log(title, title_color, message, logging.WARN)

    def error(self, title, message=""):
        """
        Logs an error message with the given title and optional message.

        Parameters:
            title (str): The title of the error message.
            message (str, optional): The optional additional message for the error. Defaults to an empty string.
        """
        self._log(title, Fore.RED, message, logging.ERROR)

    def _log(
        self,
        title: str = "",
        title_color: str = "",
        message: str = "",
        level=logging.INFO,
    ):
        """
        Logs a message with the given title and message at the specified log level.

        Parameters:
            title (str): The title of the log message. Defaults to an empty string.
            title_color (str): The color of the log message title. Defaults to an empty string.
            message (str): The log message. Defaults to an empty string.
            level (int): The log level. Defaults to logging.INFO.

        Returns:
            None
        """
        if message:
            if isinstance(message, list):
                message = " ".join(message)
        self.logger.log(
            level,
            message,
            extra={"title": str(title), "color": str(title_color)},
        )

    def set_level(self, level):
        """
        Set the level of the logger and the typing_logger.

        Args:
            level: The level to set the logger to.

        Returns:
            None
        """
        self.logger.setLevel(level)
        self.typing_logger.setLevel(level)

    def double_check(self, additionalText=None):
        """
        A function that performs a double check on the configuration.

        Parameters:
            additionalText (str, optional): Additional text to be included in the double check message.

        Returns:
            None
        """
        if not additionalText:
            additionalText = (
                "Please ensure you've setup and configured everything"
                " correctly. Read"
                " https://github.com/Torantulino/Auto-GPT#readme to"
                " double check. You can also create a github issue or"
                " join the discord and ask there!"
            )

        self.typewriter_log(
            "DOUBLE CHECK CONFIGURATION", Fore.YELLOW, additionalText
        )

    def log_json(self, data: Any, file_name: str) -> None:
        """
        Logs the given JSON data to a specified file.

        Args:
            data (Any): The JSON data to be logged.
            file_name (str): The name of the file to log the data to.

        Returns:
            None: This function does not return anything.
        """
        # Define log directory
        this_files_dir_path = os.path.dirname(__file__)
        log_dir = os.path.join(this_files_dir_path, "../logs")

        # Create a handler for JSON files
        json_file_path = os.path.join(log_dir, file_name)
        json_data_handler = JsonFileHandler(json_file_path)
        json_data_handler.setFormatter(JsonFormatter())

        # Log the JSON data using the custom file handler
        self.json_logger.addHandler(json_data_handler)
        self.json_logger.debug(data)
        self.json_logger.removeHandler(json_data_handler)

    def get_log_directory(self):
        """
        Returns the absolute path to the log directory.

        Returns:
            str: The absolute path to the log directory.
        """
        this_files_dir_path = os.path.dirname(__file__)
        log_dir = os.path.join(this_files_dir_path, "../logs")
        return os.path.abspath(log_dir)


"""
Output stream to console using simulated typing
"""


class TypingConsoleHandler(logging.StreamHandler):
    def emit(self, record):
        """
        Emit a log record to the console with simulated typing effect.

        Args:
            record (LogRecord): The log record to be emitted.

        Returns:
            None

        Raises:
            Exception: If an error occurs while emitting the log record.
        """
        min_typing_speed = 0.05
        max_typing_speed = 0.10
        # min_typing_speed = 0.005
        # max_typing_speed = 0.010

        msg = self.format(record)
        try:
            # replace enter & indent with other symbols
            transfer_enter = "<ENTER>"
            msg_transfered = str(msg).replace("\n", transfer_enter)
            transfer_space = "<4SPACE>"
            msg_transfered = str(msg_transfered).replace(
                "    ", transfer_space
            )
            words = msg_transfered.split()
            words = [word.replace(transfer_enter, "\n") for word in words]
            words = [
                word.replace(transfer_space, "    ") for word in words
            ]

            for i, word in enumerate(words):
                print(word, end="", flush=True)
                if i < len(words) - 1:
                    print(" ", end="", flush=True)
                typing_speed = random.uniform(
                    min_typing_speed, max_typing_speed
                )
                time.sleep(typing_speed)
                # type faster after each word
                min_typing_speed = min_typing_speed * 0.95
                max_typing_speed = max_typing_speed * 0.95
            print()
        except Exception:
            self.handleError(record)


class ConsoleHandler(logging.StreamHandler):
    def emit(self, record) -> None:
        """
        Emit the log record.

        Args:
            record (logging.LogRecord): The log record to emit.

        Returns:
            None: This function does not return anything.
        """
        msg = self.format(record)
        try:
            print(msg)
        except Exception:
            self.handleError(record)


class AutoGptFormatter(logging.Formatter):
    """
    Allows to handle custom placeholders 'title_color' and 'message_no_color'.
    To use this formatter, make sure to pass 'color', 'title' as log extras.
    """

    def format(self, record: LogRecord) -> str:
        """
        Formats a log record into a string representation.

        Args:
            record (LogRecord): The log record to be formatted.

        Returns:
            str: The formatted log record as a string.
        """
        if hasattr(record, "color"):
            record.title_color = (
                getattr(record, "color")
                + getattr(record, "title", "")
                + " "
                + Style.RESET_ALL
            )
        else:
            record.title_color = getattr(record, "title", "")

        # Add this line to set 'title' to an empty string if it doesn't exist
        record.title = getattr(record, "title", "")

        if hasattr(record, "msg"):
            record.message_no_color = remove_color_codes(
                getattr(record, "msg")
            )
        else:
            record.message_no_color = ""
        return super().format(record)


def remove_color_codes(s: str) -> str:
    """
    Removes color codes from a given string.

    Args:
        s (str): The string from which to remove color codes.

    Returns:
        str: The string with color codes removed.
    """
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", s)


logger = Logger()


def print_action_base(action: Action):
    """
    Print the different properties of an Action object.

    Parameters:
        action (Action): The Action object to print.

    Returns:
        None
    """
    if action.content != "":
        logger.typewriter_log("content:", Fore.YELLOW, f"{action.content}")
    logger.typewriter_log("Thought:", Fore.YELLOW, f"{action.thought}")
    if len(action.plan) > 0:
        logger.typewriter_log(
            "Plan:",
            Fore.YELLOW,
        )
        for line in action.plan:
            line = line.lstrip("- ")
            logger.typewriter_log("- ", Fore.GREEN, line.strip())
    logger.typewriter_log("Criticism:", Fore.YELLOW, f"{action.criticism}")


def print_action_tool(action: Action):
    """
    Prints the details of an action tool.

    Args:
        action (Action): The action object containing the tool details.

    Returns:
        None
    """
    logger.typewriter_log("Tool:", Fore.BLUE, f"{action.tool_name}")
    logger.typewriter_log("Tool Input:", Fore.BLUE, f"{action.tool_input}")

    output = action.tool_output if action.tool_output != "" else "None"
    logger.typewriter_log("Tool Output:", Fore.BLUE, f"{output}")

    color = Fore.RED
    if action.tool_output_status == ToolCallStatus.ToolCallSuccess:
        color = Fore.GREEN
    elif action.tool_output_status == ToolCallStatus.InputCannotParsed:
        color = Fore.YELLOW

    logger.typewriter_log(
        "Tool Call Status:",
        Fore.BLUE,
        f"{color}{action.tool_output_status.name}{Style.RESET_ALL}",
    )
