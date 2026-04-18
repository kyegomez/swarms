def exists(val):
    """Check if a value is not None.

    Args:
        val: The value to check.

    Returns:
        bool: True if val is not None, False otherwise.
    """
    return val is not None


def format_dict_to_string(data: dict, indent_level=0, use_colon=True):
    """
    Recursively format a dictionary into a multi-line string.

    Args:
        data (dict): The dictionary to format.
        indent_level (int, optional): The current indentation level for nested structures.
        use_colon (bool, optional): If True, use "key: value" formatting;
            if False, use "key value" formatting.

    Returns:
        str: Multi-line readable string representing the structure of the input dictionary.
    """
    if not isinstance(data, dict):
        return str(data)

    lines = []
    indent = "  " * indent_level
    separator = ": " if use_colon else " "

    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{indent}{key}:")
            nested_string = format_dict_to_string(
                value, indent_level + 1, use_colon
            )
            lines.append(nested_string)
        else:
            lines.append(f"{indent}{key}{separator}{value}")

    return "\n".join(lines)


def format_data_structure(
    data: any, indent_level: int = 0, max_depth: int = 10
) -> str:
    """
    Format any Python data structure into a readable, indented, multi-line string.

    Args:
        data: The data structure to format.
        indent_level (int, optional): The current indentation level. Default is 0.
        max_depth (int, optional): The maximum depth to recurse. Defaults to 10.

    Returns:
        str: Readable multi-line string representation of the input structure.
    """
    if indent_level >= max_depth:
        return f"{'  ' * indent_level}... (max depth reached)"

    indent = "  " * indent_level
    data_type = type(data)

    if data_type is dict:
        if not data:
            return f"{indent}{{}} (empty dict)"
        lines = []
        for key, value in data.items():
            if type(value) in (dict, list, tuple, set):
                lines.append(f"{indent}{key}:")
                lines.append(
                    format_data_structure(
                        value, indent_level + 1, max_depth
                    )
                )
            else:
                lines.append(f"{indent}{key}: {value}")
        return "\n".join(lines)

    elif data_type is list:
        if not data:
            return f"{indent}[] (empty list)"
        lines = []
        for i, item in enumerate(data):
            if type(item) in (dict, list, tuple, set):
                lines.append(f"{indent}[{i}]:")
                lines.append(
                    format_data_structure(
                        item, indent_level + 1, max_depth
                    )
                )
            else:
                lines.append(f"{indent}{item}")
        return "\n".join(lines)

    elif data_type is tuple:
        if not data:
            return f"{indent}() (empty tuple)"
        lines = []
        for i, item in enumerate(data):
            if type(item) in (dict, list, tuple, set):
                lines.append(f"{indent}({i}):")
                lines.append(
                    format_data_structure(
                        item, indent_level + 1, max_depth
                    )
                )
            else:
                lines.append(f"{indent}{item}")
        return "\n".join(lines)

    elif data_type is set:
        if not data:
            return f"{indent}set() (empty set)"
        lines = []
        for item in sorted(data, key=str):
            if type(item) in (dict, list, tuple, set):
                lines.append(f"{indent}set item:")
                lines.append(
                    format_data_structure(
                        item, indent_level + 1, max_depth
                    )
                )
            else:
                lines.append(f"{indent}{item}")
        return "\n".join(lines)

    elif data_type is str:
        if "\n" in data:
            lines = data.split("\n")
            return "\n".join(f"{indent}{line}" for line in lines)
        return f"{indent}{data}"

    elif data_type in (int, float, bool, type(None)):
        return f"{indent}{data}"

    else:
        if hasattr(data, "__dict__"):
            lines = [f"{indent}{data_type.__name__} object:"]
            for attr, value in data.__dict__.items():
                if not attr.startswith("_"):
                    if type(value) in (dict, list, tuple, set):
                        lines.append(f"{indent}  {attr}:")
                        lines.append(
                            format_data_structure(
                                value, indent_level + 2, max_depth
                            )
                        )
                    else:
                        lines.append(f"{indent}  {attr}: {value}")
            return "\n".join(lines)
        else:
            return f"{indent}{data} ({data_type.__name__})"
