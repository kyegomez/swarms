import json

import yaml


def remove_whitespace_from_json(json_string: str) -> str:
    """
    Removes unnecessary whitespace from a JSON string.

    This function parses the JSON string into a Python object and then
    serializes it back into a JSON string without unnecessary whitespace.

    Args:
        json_string (str): The JSON string.

    Returns:
        str: The JSON string with whitespace removed.
    """
    parsed = json.loads(json_string)
    return json.dumps(parsed, separators=(",", ":"))


# # Example usage for JSON
# json_string = '{"field1": 123, "field2": "example text"}'
# print(remove_whitespace_from_json(json_string))


def remove_whitespace_from_yaml(yaml_string: str) -> str:
    """
    Removes unnecessary whitespace from a YAML string.

    This function parses the YAML string into a Python object and then
    serializes it back into a YAML string with minimized whitespace.
    Note: This might change the representation style of YAML data.

    Args:
        yaml_string (str): The YAML string.

    Returns:
        str: The YAML string with whitespace reduced.
    """
    parsed = yaml.safe_load(yaml_string)
    return yaml.dump(parsed, default_flow_style=True)


# # Example usage for YAML
# yaml_string = """
# field1: 123
# field2: example text
# """
# print(remove_whitespace_from_yaml(yaml_string))
