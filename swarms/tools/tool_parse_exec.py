from typing import List
import json
import loguru
from swarms.utils.parse_code import extract_code_from_markdown


def parse_and_execute_json(
    functions: List[callable] = None,
    json_string: str = None,
    parse_md: bool = False,
):
    """
    Parses and executes a JSON string containing function name and parameters.

    Args:
        functions (List[callable]): A list of callable functions.
        json_string (str): The JSON string to parse and execute.
        parse_md (bool): Flag indicating whether to extract code from Markdown.

    Returns:
        The result of executing the function with the parsed parameters, or None if an error occurs.

    """
    if parse_md:
        json_string = extract_code_from_markdown(json_string)

    try:
        # Create a dictionary that maps function names to functions
        function_dict = {func.__name__: func for func in functions}

        loguru.logger.info(f"Extracted code: {json_string}")
        data = json.loads(json_string)
        function_name = data.get("function", {}).get("name")
        parameters = data.get("function", {}).get("parameters")

        # Check if the function name is in the function dictionary
        if function_name in function_dict:
            # Call the function with the parsed parameters
            result = function_dict[function_name](**parameters)
            return result
        else:
            loguru.logger.warning(
                f"No function named '{function_name}' found."
            )
            return None
    except Exception as e:
        loguru.logger.error(f"Error: {e}")
        return None
