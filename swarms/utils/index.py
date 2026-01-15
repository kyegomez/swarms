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

    Example:
        >>> format_dict_to_string({"a": 1, "b": {"c": 2}})
        'a: 1\nb:\n  c: 2'
    """
    if not isinstance(data, dict):
        return str(data)

    lines = []
    indent = "  " * indent_level  # 2 spaces per indentation level
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

    Recursively handles common container types, including dict, list, tuple, set, as well
    as custom objects with __dict__. Optionally limits recursion via max_depth.

    Args:
        data: The data structure (dict, list, tuple, set, str, int, float, bool, None, or object) to format.
        indent_level (int, optional): The current indentation level for nested structures. Default is 0.
        max_depth (int, optional): The maximum depth to recurse into nested objects. Defaults to 10.

    Returns:
        str: Readable multi-line string representation of the input structure.

    Example:
        >>> d = {"users": [{"name": "Alice", "scores": [95, 87]}, {"name": "Bob", "scores": [80]}]}
        >>> print(format_data_structure(d))
        users:
          [0]:
            name: Alice
            scores:
              [0]: 95
              [1]: 87
          [1]:
            name: Bob
            scores:
              [0]: 80
    """
    if indent_level >= max_depth:
        return f"{'  ' * indent_level}... (max depth reached)"

    indent = "  " * indent_level
    data_type = type(data)

    # Fast type checking using type() instead of isinstance() for speed
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
        for item in sorted(
            data, key=str
        ):  # Sort for consistent output
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
        # Handle multi-line strings
        if "\n" in data:
            lines = data.split("\n")
            return "\n".join(f"{indent}{line}" for line in lines)
        return f"{indent}{data}"

    elif data_type in (int, float, bool, type(None)):
        return f"{indent}{data}"

    else:
        # Handle other types (custom objects, etc.)
        if hasattr(data, "__dict__"):
            # Object with attributes
            lines = [f"{indent}{data_type.__name__} object:"]
            for attr, value in data.__dict__.items():
                if not attr.startswith(
                    "_"
                ):  # Skip private attributes
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
            # Fallback for other types
            return f"{indent}{data} ({data_type.__name__})"


# Example usage:
#
# test_dict = {
#     "name": "John",
#     "age": 30,
#     "address": {
#         "street": "123 Main St",
#         "city": "Anytown",
#         "state": "CA",
#         "zip": "12345"
#     }
# }
#
# print(format_dict_to_string(test_dict))
#
# if __name__ == "__main__":
#     # Test different data structures
#
#     # Dictionary
#     test_dict = {
#         "name": "John",
#         "age": 30,
#         "address": {
#             "street": "123 Main St",
#             "city": "Anytown"
#         }
#     }
#     print("=== Dictionary ===")
#     print(format_data_structure(test_dict))
#     print()
#
#     # List
#     test_list = ["apple", "banana", {"nested": "dict"}, [1, 2, 3]]
#     print("=== List ===")
#     print(format_data_structure(test_list))
#     print()
#
#     # Tuple
#     test_tuple = ("first", "second", {"key": "value"}, (1, 2))
#     print("=== Tuple ===")
#     print(format_data_structure(test_tuple))
#     print()
#
#     # Set
#     test_set = {"apple", "banana", "cherry"}
#     print("=== Set ===")
#     print(format_data_structure(test_set))
#     print()
#
#     # Mixed complex structure
#     complex_data = {
#         "users": [
#             {"name": "Alice", "scores": [95, 87, 92]},
#             {"name": "Bob", "scores": [88, 91, 85]}
#         ],
#         "metadata": {
#             "total_users": 2,
#             "categories": ("students", "teachers"),
#             "settings": {"debug": True, "version": "1.0"}
#         }
#     }
#     print("=== Complex Structure ===")
#     print(format_data_structure(complex_data))
