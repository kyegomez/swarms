from typing import Any


def function_to_str(function: dict[str, Any]) -> str:
    """
    Convert a function dictionary to a string representation.

    Args:
        function (dict[str, Any]): The function dictionary to convert.

    Returns:
        str: The string representation of the function.

    """
    lines = [
        f"Function: {function['name']}",
        f"Description: {function['description']}",
        "Parameters:",
    ]
    parameters = function["parameters"]["properties"]
    for param, details in parameters.items():
        lines.append(
            f"  {param} ({details['type']}): {details.get('description', '')}"
        )
    return "\n".join(lines) + "\n"


def functions_to_str(functions: list[dict[str, Any]]) -> str:
    """
    Convert a list of function dictionaries to a string representation.

    Args:
        functions (list[dict[str, Any]]): The list of function dictionaries to convert.

    Returns:
        str: The string representation of the functions.

    """
    return "\n".join(
        function_to_str(function) for function in functions
    ) + ("\n" if functions else "")
