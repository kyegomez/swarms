import re


def get_file_extension(s):
    """
    Get the file extension from a given string.

    Args:
        s (str): The input string.

    Returns:
        str or None: The file extension if found, or None if not found.

    Raises:
        ValueError: If the input is not a string.
    """
    if not isinstance(s, str):
        raise ValueError("Input must be a string")

    match = re.search(r"\.(pdf|csv|txt|docx|xlsx)$", s, re.IGNORECASE)
    return match.group()[1:] if match else None
