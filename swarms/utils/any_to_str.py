from typing import Union, Dict, List, Tuple, Any


def any_to_str(data: Union[str, Dict, List, Tuple, Any]) -> str:
    """Convert any input data type to a nicely formatted string.

    This is a thin alias for ``format_data_structure(data, style="compact")``.
    It recursively processes nested data structures and handles None values
    gracefully.

    Args:
        data: Input data of any type to convert to string. Can be:
            - Dictionary
            - List/Tuple
            - String
            - None
            - Any other type that can be converted via str()

    Returns:
        str: A formatted string representation of the input data.
            - Dictionaries are formatted as "key: value" pairs separated by newlines
            - Lists/tuples are bracket-enclosed and comma-separated
            - None returns the string "None"
            - Strings are wrapped in double quotes
            - Other types are converted using str()

    Examples:
        >>> any_to_str({'a': 1, 'b': 2})
        'a: 1\\nb: 2'
        >>> any_to_str([1, 2, 3])
        '["1", "2", "3"]'
        >>> any_to_str(None)
        'None'
    """
    from swarms.utils.index import format_data_structure

    return format_data_structure(data, style="compact")


# def main():
#     # Example 1: Dictionary
#     print("Dictionary:")
#     print(
#         any_to_str(
#             {
#                 "name": "John",
#                 "age": 30,
#                 "hobbies": ["reading", "hiking"],
#             }
#         )
#     )
#
#     print("\nNested Dictionary:")
#     print(
#         any_to_str(
#             {
#                 "user": {
#                     "id": 123,
#                     "details": {"city": "New York", "active": True},
#                 },
#                 "data": [1, 2, 3],
#             }
#         )
#     )
#
#     print("\nList and Tuple:")
#     print(any_to_str([1, "text", None, (1, 2)]))
#     print(any_to_str((True, False, None)))
#
#     print("\nEmpty Collections:")
#     print(any_to_str([]))
#     print(any_to_str({}))
#
#
# if __name__ == "__main__":
#     main()
