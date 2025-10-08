"""
Custom docstring parser implementation to replace the docstring_parser package.

This module provides a simple docstring parser that extracts parameter information
and descriptions from Python docstrings in Google/NumPy style format.
"""

import re
from typing import List, Optional, NamedTuple


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

    # Clean up the docstring
    lines = [line.strip() for line in docstring.strip().split("\n")]

    # Extract short description (first non-empty line that's not a section header)
    short_description = None
    for line in lines:
        if line and not line.startswith(
            (
                "Args:",
                "Parameters:",
                "Returns:",
                "Yields:",
                "Raises:",
                "Note:",
                "Example:",
                "Examples:",
            )
        ):
            short_description = line
            break

    # Extract parameters
    params = []

    # Look for Args: or Parameters: section
    in_args_section = False
    current_param = None

    for line in lines:
        # Check if we're entering the Args/Parameters section
        if line.lower().startswith(("args:", "parameters:")):
            in_args_section = True
            continue

        # Check if we're leaving the Args/Parameters section
        if (
            in_args_section
            and line
            and not line.startswith(" ")
            and not line.startswith("\t")
        ):
            # Check if this is a new section header
            if line.lower().startswith(
                (
                    "returns:",
                    "yields:",
                    "raises:",
                    "note:",
                    "example:",
                    "examples:",
                    "see also:",
                    "see_also:",
                )
            ):
                in_args_section = False
                if current_param:
                    params.append(current_param)
                    current_param = None
                continue

        if in_args_section and line:
            # Check if this line starts a new parameter (starts with parameter name)
            # Pattern: param_name (type): description
            param_match = re.match(
                r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)$", line
            )
            if param_match:
                # Save previous parameter if exists
                if current_param:
                    params.append(current_param)

                param_name = param_match.group(1)
                param_desc = param_match.group(2).strip()
                current_param = DocstringParam(
                    arg_name=param_name, description=param_desc
                )
            elif current_param and (
                line.startswith(" ") or line.startswith("\t")
            ):
                # This is a continuation of the current parameter description
                current_param = DocstringParam(
                    arg_name=current_param.arg_name,
                    description=current_param.description
                    + " "
                    + line.strip(),
                )
            elif not line.startswith(" ") and not line.startswith(
                "\t"
            ):
                # This might be a new section, stop processing args
                in_args_section = False
                if current_param:
                    params.append(current_param)
                    current_param = None

    # Add the last parameter if it exists
    if current_param:
        params.append(current_param)

    return DocstringInfo(
        short_description=short_description, params=params
    )
