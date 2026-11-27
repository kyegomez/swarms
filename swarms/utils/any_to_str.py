from typing import Union, Dict, List, Tuple, Any


def any_to_str(data: Union[str, Dict, List, Tuple, Any]) -> str:
    """Format a value as text, recursively formatting nested containers.

    Args:
        data: The value to format.

    Returns:
        A string with one key-value pair per line for dictionaries,
        comma-separated items in brackets for lists or parentheses for
        tuples, and double quotes around strings. None becomes "None".
        Other values are converted with str().

    Examples:
        >>> print(any_to_str({'a': 1, 'b': 2}))
        a: 1
        b: 2
        >>> any_to_str([1, 2, 3])
        '[1, 2, 3]'
        >>> any_to_str(None)
        'None'
    """
    try:
        if isinstance(data, dict):
            # Format dictionary with newlines and indentation
            items = []
            for k, v in data.items():
                value = any_to_str(v)
                items.append(f"{k}: {value}")
            return "\n".join(items)

        elif isinstance(data, (list, tuple)):
            # Format sequences with brackets and proper spacing
            items = [any_to_str(x) for x in data]
            if len(items) == 0:
                return "[]" if isinstance(data, list) else "()"
            return (
                f"[{', '.join(items)}]"
                if isinstance(data, list)
                else f"({', '.join(items)})"
            )

        elif data is None:
            return "None"

        else:
            # Handle strings and other types
            if isinstance(data, str):
                return f'"{data}"'
            return str(data)

    except Exception as e:
        return f"Error converting data: {str(e)}"
