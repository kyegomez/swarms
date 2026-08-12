from typing import Any, List

from swarms.utils.docstring_parser import parse
from pydantic import BaseModel


def _remove_a_key(d: dict, remove_key: str) -> None:
    """
    Recursively remove a specified key from a nested dictionary.

    Args:
        d (dict): The dictionary from which to remove the key.
        remove_key (str): The key to remove from the dictionary.

    Returns:
        None: The provided dictionary is modified in-place.
    """
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == remove_key and "type" in d.keys():
                del d[key]
            else:
                _remove_a_key(d[key], remove_key)


def base_model_to_openai_function(
    pydantic_type: type[BaseModel],
    output_str: bool = False,
) -> dict[str, Any]:
    """
    Convert a Pydantic model class to an OpenAI-compatible function specification dictionary.

    This function takes a Pydantic model type, extracts its schema, properties, and docstring,
    and returns a dictionary representation suitable for use with OpenAI's function-calling API.
    Optionally, it can output the result as a formatted JSON string.

    Args:
        pydantic_type (type[BaseModel]): The Pydantic model class to convert.
        output_str (bool, optional): If True, returns a pretty-printed JSON string. Defaults to False.

    Returns:
        dict[str, Any]: The OpenAI-compatible function specification, or a JSON string if output_str is True.
    """
    schema = pydantic_type.model_json_schema()

    # The model's own name. `type(pydantic_type).__name__` read the
    # metaclass instead, because pydantic_type is the class, not an
    # instance — every schema through this path was named "ModelMetaclass".
    name = pydantic_type.__name__

    docstring = parse(pydantic_type.__doc__ or "")
    parameters = {
        k: v
        for k, v in schema.items()
        if k not in ("title", "description")
    }

    # `prop_name`, not `name`: this walrus used to rebind `name`, so the
    # emitted function name became whichever docstring param matched last.
    for param in docstring.params:
        if (prop_name := param.arg_name) in parameters[
            "properties"
        ] and (description := param.description):
            if (
                "description"
                not in parameters["properties"][prop_name]
            ):
                parameters["properties"][prop_name][
                    "description"
                ] = description

    parameters["type"] = "object"

    if "description" not in schema:
        if docstring.short_description:
            schema["description"] = docstring.short_description
        else:
            schema["description"] = (
                f"Correctly extracted `{name}` with all "
                f"the required parameters with correct types"
            )

    _remove_a_key(parameters, "title")
    _remove_a_key(parameters, "additionalProperties")

    result = {
        "function_call": {
            "name": name,
        },
        "functions": [
            {
                "name": name,
                "description": schema["description"],
                "parameters": parameters,
            },
        ],
    }

    # Handle output_str parameter
    if output_str:
        import json

        return json.dumps(result, indent=2)

    return result


def multi_base_model_to_openai_function(
    pydantic_types: List[BaseModel] = None,
    output_str: bool = False,
) -> dict[str, Any]:
    """
    Convert multiple Pydantic model classes to a combined OpenAI function specification.

    This function takes a list of Pydantic types and returns a dictionary containing
    all their OpenAI-compatible function specifications, with a default "auto" function_call.
    Optionally returns the output as a pretty-printed JSON string.

    Args:
        pydantic_types (List[BaseModel]): A list of Pydantic model classes to convert.
        output_str (bool, optional): If True, outputs a formatted JSON string. Defaults to False.

    Returns:
        dict[str, Any]: Dictionary with combined OpenAI function specifications,
            or a JSON string if output_str is True.
    """
    functions: list[dict[str, Any]] = [
        base_model_to_openai_function(
            pydantic_type, output_str=False
        )["functions"][0]
        for pydantic_type in pydantic_types
    ]

    result = {
        "function_call": "auto",
        "functions": functions,
    }

    # Handle output_str parameter
    if output_str:
        import json

        return json.dumps(result, indent=2)

    return result
