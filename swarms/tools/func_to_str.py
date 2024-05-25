from typing import Any


def function_to_str(function: dict[str, Any]) -> str:
    """
    Convert a function dictionary to a string representation.

    Args:
        function (dict[str, Any]): The function dictionary to convert.

    Returns:
        str: The string representation of the function.

    """
    function_str = f"Function: {function['name']}\n"
    function_str += f"Description: {function['description']}\n"
    function_str += "Parameters:\n"

    for param, details in function["parameters"]["properties"].items():
        function_str += f"  {param} ({details['type']}): {details.get('description', '')}\n"

    return function_str


def functions_to_str(functions: list[dict[str, Any]]) -> str:
    """
    Convert a list of function dictionaries to a string representation.

    Args:
        functions (list[dict[str, Any]]): The list of function dictionaries to convert.

    Returns:
        str: The string representation of the functions.

    """
    functions_str = ""
    for function in functions:
        functions_str += function_to_str(function) + "\n"

    return functions_str
