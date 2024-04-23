from typing import Any, Optional, List

from docstring_parser import parse
from pydantic import BaseModel


def _remove_a_key(d: dict, remove_key: str) -> None:
    """Remove a key from a dictionary recursively"""
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == remove_key and "type" in d.keys():
                del d[key]
            else:
                _remove_a_key(d[key], remove_key)


def pydantic_to_functions(
    pydantic_type: type[BaseModel],
) -> dict[str, Any]:
    """
    Convert a Pydantic model to a dictionary representation of functions.

    Args:
        pydantic_type (type[BaseModel]): The Pydantic model type to convert.

    Returns:
        dict[str, Any]: A dictionary representation of the functions.

    """
    schema = pydantic_type.model_json_schema()

    docstring = parse(pydantic_type.__doc__ or "")
    parameters = {
        k: v
        for k, v in schema.items()
        if k not in ("title", "description")
    }

    for param in docstring.params:
        if (name := param.arg_name) in parameters["properties"] and (
            description := param.description
        ):
            if "description" not in parameters["properties"][name]:
                parameters["properties"][name]["description"] = description

    parameters["type"] = "object"

    if "description" not in schema:
        if docstring.short_description:
            schema["description"] = docstring.short_description
        else:
            schema["description"] = (
                f"Correctly extracted `{pydantic_type.__class__.__name__.lower()}` with all "
                f"the required parameters with correct types"
            )

    _remove_a_key(parameters, "title")
    _remove_a_key(parameters, "additionalProperties")

    return {
        "function_call": {
            "name": pydantic_type.__class__.__name__.lower(),
        },
        "functions": [
            {
                "name": pydantic_type.__class__.__name__.lower(),
                "description": schema["description"],
                "parameters": parameters,
            },
        ],
    }


def multi_pydantic_to_functions(
    pydantic_types: List[BaseModel] = None
) -> dict[str, Any]:
    """
    Converts multiple Pydantic types to a dictionary of functions.

    Args:
        pydantic_types (List[BaseModel]]): A list of Pydantic types to convert.

    Returns:
        dict[str, Any]: A dictionary containing the converted functions.

    """
    functions: list[dict[str, Any]] = [
        pydantic_to_functions(pydantic_type)["functions"][0]
        for pydantic_type in pydantic_types
    ]

    return {
        "function_call": "auto",
        "functions": functions,
    }


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
