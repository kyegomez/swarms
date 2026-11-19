import json
from typing import Dict


def str_to_dict(s: str) -> Dict:
    """
    Converts a JSON string to dictionary.

    Args:
        s (str): The JSON string to be converted.

    Returns:
        Dict: The parsed dictionary from the JSON string.

    Raises:
        json.JSONDecodeError: If the string cannot be parsed.
    """
    return json.loads(s)
