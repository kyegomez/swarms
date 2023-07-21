import re
from typing import Dict

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