from swarms.tools.tool_parse_exec import parse_and_execute_json


# Example functions to be called
def add(a: int, b: int) -> int:
    """
    Adds two integers and returns the result.

    Parameters:
    a (int): The first integer.
    b (int): The second integer.

    Returns:
    int: The sum of the two integers.
    """
    return a + b


def subtract(a: int, b: int) -> int:
    """
    Subtracts two integers and returns the result.

    Parameters:
    a (int): The first integer.
    b (int): The second integer.

    Returns:
    int: The difference between the two integers.
    """
    return a - b


def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers.

    Args:
        a (int): The first number.
        b (int): The second number.

    Returns:
        int: The product of the two numbers.
    """
    return a * b


# Example usage
functions_list = [add, subtract, multiply]
json_input = """
{
    "function": [
        {"name": "add", "parameters": {"a": 10, "b": 5}},
        {"name": "subtract", "parameters": {"a": 10, "b": 5}},
        {"name": "multiply", "parameters": {"a": 10, "b": 5}}
    ]
}
"""


json_input_single = """
{
    "function": {"name": "add", "parameters": {"a": 10, "b": 5}}
}
"""


# Testing multiple functions
results_multiple = parse_and_execute_json(
    functions=functions_list,
    json_string=json_input,
    parse_md=False,
    verbose=True,
)
print("Multiple functions results:\n", results_multiple)

# Testing single function
results_single = parse_and_execute_json(
    functions=functions_list,
    json_string=json_input_single,
    parse_md=False,
    verbose=True,
)
print("Single function result:\n", results_single)
