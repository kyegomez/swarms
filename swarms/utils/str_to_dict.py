import json
from typing import Dict


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
