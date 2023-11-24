import re

# def extract_code_in_backticks_in_string(s: str) -> str:
# """
# Extracts code blocks from a markdown string.

# Args:
#     s (str): The markdown string to extract code from.

# Returns:
#     list: A list of tuples. Each tuple contains the language of the code block (if specified) and the code itself.
# """
# pattern = r"```([\w\+\#\-\.\s]*)\n(.*?)```"
# matches = re.findall(pattern, s, re.DOTALL)
# out =  [(match[0], match[1].strip()) for match in matches]
# print(out)


def extract_code_in_backticks_in_string(s: str) -> str:
    """
<<<<<<< HEAD
    Extracts code blocks from a markdown string.

    Args:
        s (str): The markdown string to extract code from.

    Returns:
        str: A string containing all the code blocks.
    """
    pattern = r"```([\w\+\#\-\.\s]*)(.*?)```"
    matches = re.findall(pattern, s, re.DOTALL)
    return "\n".join(match[1].strip() for match in matches)
=======
    pattern = r"`` ``(.*?)`` "  # Non-greedy match between six backticks
    match = re.search(
        pattern, message, re.DOTALL
    )  # re.DOTALL to match newline chars
    return match.group(1).strip() if match else None
>>>>>>> 49c7b97c (code quality fixes: line length = 80)
