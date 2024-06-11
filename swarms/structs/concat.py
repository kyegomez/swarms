from typing import List


def concat_strings(string_list: List[str]) -> str:
    """
    Concatenates a list of strings into a single string.

    Args:
        string_list (List[str]): A list of strings to be concatenated.

    Returns:
        str: The concatenated string.

    Raises:
        TypeError: If the input is not a list of strings.

    """
    if not isinstance(string_list, list):
        raise TypeError("Input must be a list of strings.")

    try:
        return "".join(string_list)
    except TypeError:
        raise TypeError("All elements in the list must be strings.")
