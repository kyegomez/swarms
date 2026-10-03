import ast
import json
from typing import Any, Dict, Optional


def str_to_dict(s: str, retries: int = 3) -> Dict:
    """
    Converts a JSON string to dictionary.

    Args:
        s (str): The JSON string to be converted.
        retries (int): The number of times to retry parsing the string in case of a JSONDecodeError. Default is 3.

    Returns:
        Dict: The parsed dictionary from the JSON string.

    Raises:
        json.JSONDecodeError: If the string cannot be parsed into a dictionary after the specified number of retries.
    """
    for attempt in range(retries):
        try:
            # Run json.loads directly since it's fast enough
            return json.loads(s)
        except json.JSONDecodeError as e:
            if attempt < retries - 1:
                continue  # Retry on failure
            else:
                raise e  # Raise the error if all retries fail


def tool_call_arguments(tool_output: Any) -> Optional[Dict]:
    """
    Unwrap a forced tool call into its ``arguments`` mapping.

    Providers hand this structure back three different ways, and all three
    are accepted here so every caller sees the same thing:

    - a plain dict, ``{"function": {"arguments": ...}}``
    - a pydantic object whose ``function`` and ``arguments`` are attributes
    - the ``repr`` of a list of either, which is what an agent whose
      ``output_type`` renders to text produces

    Args:
        tool_output (Any): Raw provider output from a forced tool call.

    Returns:
        Optional[Dict]: The parsed ``arguments`` mapping, or None when the
        output is missing, unparseable, or carries no arguments dict. Callers
        supply their own defaults for the None case.
    """
    if isinstance(tool_output, str):
        try:
            tool_output = ast.literal_eval(tool_output)
        except (ValueError, SyntaxError):
            return None

    if isinstance(tool_output, list):
        tool_output = tool_output[0] if tool_output else None
    if not tool_output:
        return None

    if not isinstance(tool_output, dict) and hasattr(
        tool_output, "model_dump"
    ):
        try:
            tool_output = tool_output.model_dump()
        except Exception:
            pass

    if isinstance(tool_output, dict):
        fn = tool_output.get("function")
    else:
        fn = getattr(tool_output, "function", None)
    if not fn:
        return None

    if isinstance(fn, dict):
        args = fn.get("arguments")
    else:
        args = getattr(fn, "arguments", None)

    if isinstance(args, (str, bytes, bytearray)):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    return args if isinstance(args, dict) else None
