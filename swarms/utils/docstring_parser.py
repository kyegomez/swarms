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
_NUMPY_PARAM_RE = re.compile(r"^(\w+)(?:\s*:\s*.+)?$")
_ARG_HEADERS = ("args:", "arguments:", "parameters:")


class DocstringParam(NamedTuple):
    """Represents a parameter in a docstring."""

    arg_name: str
    description: str


class DocstringInfo(NamedTuple):
    """Represents parsed docstring information."""

    short_description: Optional[str]
    params: List[DocstringParam]
    description: Optional[str]


def parse(docstring: str) -> DocstringInfo:
    """
    Parse a docstring and extract parameter information and description.

    Args:
        docstring (str): The docstring to parse.

    Returns:
        DocstringInfo: Parsed docstring information containing short description and parameters.
    """
    if not docstring or not docstring.strip():
        return DocstringInfo(
            short_description=None, params=[], description=None
        )

    # cleandoc keeps the relative indentation that separates a wrapped line from the next parameter
    lines = inspect.cleandoc(docstring).split("\n")
    numpy_headers = {
        index: f"{line.rstrip().lower()}:"
        for index, line in enumerate(lines[:-1])
        if f"{line.rstrip().lower()}:" in _SECTION_HEADERS
        and re.fullmatch(r"-+", lines[index + 1].rstrip())
    }

    prose = []
    for index, line in enumerate(lines):
        if numpy_headers.get(index, line.strip().lower()).startswith(
            _SECTION_HEADERS
        ):
            break
        prose.append(line)
    description = "\n".join(prose).strip() or None
    short_description = (
        description.split("\n")[0].strip() if description else None
    )

    # Extract parameters
    params: List[DocstringParam] = []
    in_args_section = False
    numpy_args = False
    current_param = None
    param_indent = 0

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        header = numpy_headers.get(index, stripped.lower())

        if not in_args_section:
            if header.startswith(_ARG_HEADERS):
                in_args_section = True
                numpy_args = index in numpy_headers
            continue

        # Checked before the parameter pattern, which a wrapped line with a colon would match
        if current_param is not None and indent > param_indent:
            current_param = DocstringParam(
                arg_name=current_param.arg_name,
                description=f"{current_param.description} {stripped}".lstrip(),
            )
            continue

        if header.startswith(_SECTION_HEADERS):
            break

        pattern = _NUMPY_PARAM_RE if numpy_args else _PARAM_RE
        param_match = pattern.match(stripped)
        if param_match:
            if current_param:
                params.append(current_param)
            current_param = DocstringParam(
                arg_name=param_match.group(1),
                description=(
                    "" if numpy_args else param_match.group(2).strip()
                ),
            )
            param_indent = indent

    if current_param:
        params.append(current_param)

    return DocstringInfo(
        short_description=short_description,
        params=params,
        description=description,
    )
