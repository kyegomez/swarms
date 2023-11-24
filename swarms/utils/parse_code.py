import re


def extract_code_in_backticks_in_string(message: str) -> str:
    """
    To extract code from a string in markdown and return a string

    """
    pattern = r"`` ``(.*?)`` "  # Non-greedy match between six backticks
    match = re.search(
        pattern, message, re.DOTALL
    )  # re.DOTALL to match newline chars
    return match.group(1).strip() if match else None
