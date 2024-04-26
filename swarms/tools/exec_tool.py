import json
import concurrent.futures
import re
from abc import abstractmethod
from typing import Dict, List, NamedTuple

from langchain.schema import BaseOutputParser
from pydantic import ValidationError

from swarms.tools.tool import BaseTool

from swarms.utils.loguru_logger import logger


class AgentAction(NamedTuple):
    """Action returned by AgentOutputParser."""

    name: str
    args: Dict


class BaseAgentOutputParser(BaseOutputParser):
    """Base Output parser for Agent."""

    @abstractmethod
    def parse(self, text: str) -> AgentAction:
        """Return AgentAction"""


def preprocess_json_input(input_str: str) -> str:
    """Preprocesses a string to be parsed as json.

    Replace single backslashes with double backslashes,
    while leaving already escaped ones intact.

    Args:
        input_str: String to be preprocessed

    Returns:
        Preprocessed string
    """
    corrected_str = re.sub(
        r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})',
        r"\\\\",
        input_str,
    )
    return corrected_str


class AgentOutputParser(BaseAgentOutputParser):
    """Output parser for Agent."""

    def parse(self, text: str) -> AgentAction:
        try:
            parsed = json.loads(text, strict=False)
        except json.JSONDecodeError:
            preprocessed_text = preprocess_json_input(text)
            try:
                parsed = json.loads(preprocessed_text, strict=False)
            except Exception:
                return AgentAction(
                    name="ERROR",
                    args={
                        "error": (f"Could not parse invalid json: {text}")
                    },
                )
        try:
            return AgentAction(
                name=parsed["command"]["name"],
                args=parsed["command"]["args"],
            )
        except (KeyError, TypeError):
            # If the command is null or incomplete, return an erroneous tool
            return AgentAction(
                name="ERROR",
                args={"error": f"Incomplete command args: {parsed}"},
            )


def execute_tool_by_name(
    text: str,
    tools: List[BaseTool],
    stop_token: str = "finish",
):
    """
    Executes a tool based on the given text command.

    Args:
        text (str): The text command to be executed.
        tools (List[BaseTool]): A list of available tools.
        stop_token (str, optional): The stop token to terminate the execution. Defaults to "finish".

    Returns:
        str: The result of the command execution.
    """
    output_parser = AgentOutputParser()
    # Get command name and arguments
    action = output_parser.parse(text)
    tools = {t.name: t for t in tools}

    # logger.info(f"Tools available: {tools}")

    if action.name == stop_token:
        return action.args["response"]
    if action.name in tools:
        tool = tools[action.name]
        try:
            # Check if multiple tools are used
            tool_names = [name for name in tools if name in text]
            if len(tool_names) > 1:
                # Execute tools concurrently
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = []
                    for tool_name in tool_names:
                        logger.info(f"Executing tool: {tool_name}")
                        futures.append(
                            executor.submit(
                                tools[tool_name].run, action.args
                            )
                        )

                    # Wait for all futures to complete
                    concurrent.futures.wait(futures)

                    # Get results from completed futures
                    results = [
                        future.result()
                        for future in futures
                        if future.done()
                    ]

                    # Process results
                    for result in results:
                        # Handle errors
                        if isinstance(result, Exception):
                            result = (
                                f"Error: {str(result)},"
                                f" {type(result).__name__}, args:"
                                f" {action.args}"
                            )
                        # Handle successful execution
                        else:
                            result = (
                                f"Command {tool.name} returned:"
                                f" {result}"
                            )
            else:
                observation = tool.run(action.args)
        except ValidationError as e:
            observation = (
                f"Validation Error in args: {str(e)}, args:"
                f" {action.args}"
            )
        except Exception as e:
            observation = (
                f"Error: {str(e)}, {type(e).__name__}, args:"
                f" {action.args}"
            )
        result = f"Command {tool.name} returned: {observation}"
    elif action.name == "ERROR":
        result = f"Error: {action.args}. "
    else:
        result = (
            f"Unknown command '{action.name}'. "
            "Please refer to the 'COMMANDS' list for available "
            "commands and only respond in the specified JSON format."
        )
    return result
