from typing import Union, Dict, List, Tuple, Any
from swarms.utils.index import format_data_structure

def any_to_str(data: Union[str, Dict, List, Tuple, Any]) -> str:
    """Convert any input data type to a nicely formatted string.

    This function handles conversion of various Python data types into a clean string representation.
    It recursively processes nested data structures and handles None values gracefully.

    Args:
        data: Input data of any type to convert to string. Can be:
            - Dictionary
            - List/Tuple
            - String
            - None
            - Any other type that can be converted via str()

    Returns:
        str: A formatted string representation of the input data.
            - Dictionaries are formatted as "key: value" pairs separated by commas
            - Lists/tuples are comma-separated
            - None returns empty string
            - Other types are converted using str()

    Examples:
        >>> any_to_str({'a': 1, 'b': 2})
        'a: 1, b: 2'
        >>> any_to_str([1, 2, 3])
        '1, 2, 3'
        >>> any_to_str(None)
        ''
    """
    return format_data_structure(data, style="compact")
