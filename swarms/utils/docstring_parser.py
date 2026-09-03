"""
Custom docstring parser implementation to replace the docstring_parser package.

This module provides a simple docstring parser that extracts parameter information
and descriptions from Python docstrings in Google/NumPy style format.
"""

import inspect
import re
from typing import List, Optional, NamedTuple

# Headers that end the description and, inside Args, end the parameter list.
_SECTION_HEADERS = (
    "args:",
    "arguments:",
    "parameters:",
    "returns:",
    "yields:",
    "raises:",
    "note:",
    "notes:",
    "example:",
    "examples:",
    "see also:",
    "see_also:",
    "attributes:",
)

_PARAM_RE = re.compile(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)$")


class DocstringParam(NamedTuple):
    """Represents a parameter in a docstring."""

    arg_name: str
    description: str


class DocstringInfo(NamedTuple):
    """Represents parsed docstring information."""

    short_description: Optional[str]
    params: List[DocstringParam]


def parse(docstring: str) -> DocstringInfo:
    """
    Parse a docstring and extract parameter information and description.

    Args:
        docstring (str): The docstring to parse.

    Returns:
        DocstringInfo: Parsed docstring information containing short description and parameters.
    """
    if not docstring or not docstring.strip():
        return DocstringInfo(short_description=None, params=[])

    # cleandoc, not strip-every-line: the parser below needs the relative
    # indentation to tell a wrapped description from the next parameter.
    lines = inspect.cleandoc(docstring).split("\n")

    # Extract short description: the first prose line before any section.
    short_description = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith(_SECTION_HEADERS):
            break
        short_description = stripped
        break

    # Extract parameters
    params: List[DocstringParam] = []
    in_args_section = False
    current_param = None
    param_indent = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        lowered = stripped.lower()

        if not in_args_section:
            if lowered.startswith(
                ("args:", "arguments:", "parameters:")
            ):
                in_args_section = True
            continue

        # A continuation is indented past the parameter it belongs to. Check
        # this before the parameter pattern, which a wrapped line containing a
        # colon would otherwise match.
        if current_param is not None and indent > param_indent:
            current_param = DocstringParam(
                arg_name=current_param.arg_name,
                description=f"{current_param.description} {stripped}",
            )
            continue

        if lowered.startswith(_SECTION_HEADERS):
            break

        param_match = _PARAM_RE.match(stripped)
        if param_match:
            if current_param:
                params.append(current_param)
            current_param = DocstringParam(
                arg_name=param_match.group(1),
                description=param_match.group(2).strip(),
            )
            param_indent = indent

    if current_param:
        params.append(current_param)

    return DocstringInfo(
        short_description=short_description, params=params
    )
