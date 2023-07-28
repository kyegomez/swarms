import json
import re
from abc import abstractmethod
from typing import Dict, NamedTuple

from langchain.schema import BaseOutputParser

from swarms.agents.prompts.prompts import EVAL_FORMAT_INSTRUCTIONS


class EvalOutputParser(BaseOutputParser):
    @staticmethod
    def parse_all(text: str) -> Dict[str, str]:
        regex = r"Action: (.*?)[\n]Plan:(.*)[\n]What I Did:(.*)[\n]Action Input: (.*)"
        match = re.search(regex, text, re.DOTALL)
        if not match:
            raise Exception("parse error")

        action = match.group(1).strip()
        plan = match.group(2)
        what_i_did = match.group(3)
        action_input = match.group(4).strip(" ")

        return {
            "action": action,
            "plan": plan,
            "what_i_did": what_i_did,
            "action_input": action_input,
        }

    def get_format_instructions(self) -> str:
        return EVAL_FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Dict[str, str]:
        regex = r"Action: (.*?)[\n]Plan:(.*)[\n]What I Did:(.*)[\n]Action Input: (.*)"
        match = re.search(regex, text, re.DOTALL)
        if not match:
            raise Exception("parse error")

        parsed = EvalOutputParser.parse_all(text)

        return {"action": parsed["action"], "action_input": parsed["action_input"]}

    def __str__(self):
        return "EvalOutputParser"
    


class AgentAction(NamedTuple):
    """Action for Agent."""

    name: str
    """Name of the action."""
    args: Dict
    """Arguments for the action."""


class BaseAgentOutputParser(BaseOutputParser):
    """Base class for Agent output parsers."""

    @abstractmethod
    def parse(self, text: str) -> AgentAction:
        """Parse text and return AgentAction"""


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
        r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r"\\\\", input_str
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
                    args={"error": f"Could not parse invalid json: {text}"},
                )
        try:
            return AgentAction(
                name=parsed["command"]["name"],
                args=parsed["command"]["args"],
            )
        except (KeyError, TypeError):
            # If the command is null or incomplete, return an erroneous tool
            return AgentAction(
                name="ERROR", args={"error": f"Incomplete command args: {parsed}"}
            )