import json
import re
from abc import abstractmethod
from typing import Dict, NamedTuple


class AgentAction(NamedTuple):
    """Action returned by AgentOutputParser."""

    name: str
    args: Dict


class BaseAgentOutputParser:
    """Base Output parser for Agent."""

    @abstractmethod
    def parse(self, text: str) -> AgentAction:
        """Return AgentAction"""


class AgentOutputParser(BaseAgentOutputParser):
    """Output parser for Agent."""

    @staticmethod
    def _preprocess_json_input(input_str: str) -> str:
        corrected_str = re.sub(
            r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})',
            r"\\\\",
            input_str,
        )
        return corrected_str

    def _parse_json(self, text: str) -> dict:
        try:
            parsed = json.loads(text, strict=False)
        except json.JSONDecodeError:
            preprocessed_text = self._preprocess_json_input(text)
            parsed = json.loads(preprocessed_text, strict=False)
        return parsed

    def parse(self, text: str) -> AgentAction:
        try:
            parsed = self._parse_json(text)
            return AgentAction(
                name=parsed["command"]["name"],
                args=parsed["command"]["args"],
            )
        except (KeyError, TypeError, json.JSONDecodeError) as e:
            return AgentAction(
                name="ERROR",
                args={"error": f"Error in parsing: {e}"},
            )
