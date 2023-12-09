import requests
import json
from ..tool import Tool
import os

import sys
from io import StringIO
from typing import Dict, Optional


class PythonREPL:
    """Simulates a standalone Python REPL."""

    def __init__(self) -> None:
        self.globals: Optional[Dict] = globals()
        self.locals: Optional[Dict] = None

    def run(self, command: str) -> str:
        """Run command with own globals/locals and returns anything printed."""
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        try:
            exec(command, self.globals, self.locals)
            sys.stdout = old_stdout
            output = mystdout.getvalue()
        except Exception as e:
            sys.stdout = old_stdout
            output = repr(e)
        print(output)
        return output


def build_tool(config) -> Tool:
    tool = Tool(
        "Python REPL",
        "Run python code",
        name_for_model="Python REPL",
        description_for_model=(
            "A Python shell. Use this to execute python commands. "
            "Input should be a valid python command. "
            "If you want to see the output of a value, you should print it out "
            "with `print(...)`."
        ),
        logo_url="https://your-app-url.com/.well-known/logo.png",
        contact_email="hello@contact.com",
        legal_info_url="hello@legal.com",
    )

    python_repl = PythonREPL()
    sanitize_input: bool = True

    @tool.get("/run_python")
    def run_python(query: str):
        """Run python code in a REPL."""
        if sanitize_input:
            query = query.strip().strip("```")
        return python_repl.run(query)

    return tool
