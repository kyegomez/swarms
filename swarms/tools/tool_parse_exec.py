import json
from typing import List

from swarms.utils.loguru_logger import logger
from swarms.utils.parse_code import extract_code_from_markdown


def parse_and_execute_json(
    functions: List[callable] = None,
    json_string: str = None,
    parse_md: bool = False,
):
    """
    Parses and executes a JSON string containing function names and parameters.

    Args:
        functions (List[callable]): A list of callable functions.
        json_string (str): The JSON string to parse and execute.
        parse_md (bool): Flag indicating whether to extract code from Markdown.

    Returns:
        A dictionary containing the results of executing the functions with the parsed parameters.

    """
    if parse_md:
        json_string = extract_code_from_markdown(json_string)

    try:
        # Create a dictionary that maps function names to functions
        function_dict = {func.__name__: func for func in functions}

        data = json.loads(json_string)
        function_list = (
            data.get("functions", [])
            if data.get("functions")
            else [data.get("function", [])]
        )

        results = {}
        for function_data in function_list:
            function_name = function_data.get("name")
            parameters = function_data.get("parameters")

            # Check if the function name is in the function dictionary
            if function_name in function_dict:
                # Call the function with the parsed parameters
                result = function_dict[function_name](**parameters)
                results[function_name] = str(result)
            else:
                results[function_name] = None

        return results
    except Exception as e:
        logger.error(f"Error parsing and executing JSON: {e}")
        return None


# def parse_and_execute_json(
#     functions: List[Callable[..., Any]],
#     json_string: str = None,
#     parse_md: bool = False,
#     verbose: bool = False,
# ) -> Dict[str, Any]:
#     """
#     Parses and executes a JSON string containing function names and parameters.

#     Args:
#         functions (List[Callable]): A list of callable functions.
#         json_string (str): The JSON string to parse and execute.
#         parse_md (bool): Flag indicating whether to extract code from Markdown.
#         verbose (bool): Flag indicating whether to enable verbose logging.

#     Returns:
#         Dict[str, Any]: A dictionary containing the results of executing the functions with the parsed parameters.
#     """
#     if parse_md:
#         json_string = extract_code_from_markdown(json_string)

#     logger.info("Number of functions: " + str(len(functions)))

#     try:
#         # Create a dictionary that maps function names to functions
#         function_dict = {func.__name__: func for func in functions}

#         data = json.loads(json_string)
#         function_list = data.get("functions") or [data.get("function")]

#         # Ensure function_list is a list and filter out None values
#         if isinstance(function_list, dict):
#             function_list = [function_list]
#         else:
#             function_list = [f for f in function_list if f]

#         results = {}

#         # Determine if concurrency is needed
#         concurrency = len(function_list) > 1

#         if concurrency:
#             with concurrent.futures.ThreadPoolExecutor() as executor:
#                 future_to_function = {
#                     executor.submit(
#                         execute_and_log_function,
#                         function_dict,
#                         function_data,
#                         verbose,
#                     ): function_data
#                     for function_data in function_list
#                 }
#                 for future in concurrent.futures.as_completed(
#                     future_to_function
#                 ):
#                     function_data = future_to_function[future]
#                     try:
#                         result = future.result()
#                         results.update(result)
#                     except Exception as e:
#                         if verbose:
#                             logger.error(
#                                 f"Error executing function {function_data.get('name')}: {e}"
#                             )
#                         results[function_data.get("name")] = None
#         else:
#             for function_data in function_list:
#                 function_name = function_data.get("name")
#                 parameters = function_data.get("parameters")

#                 if verbose:
#                     logger.info(
#                         f"Executing function: {function_name} with parameters: {parameters}"
#                     )

#                 if function_name in function_dict:
#                     try:
#                         result = function_dict[function_name](**parameters)
#                         results[function_name] = str(result)
#                         if verbose:
#                             logger.info(
#                                 f"Result for function {function_name}: {result}"
#                             )
#                     except Exception as e:
#                         if verbose:
#                             logger.error(
#                                 f"Error executing function {function_name}: {e}"
#                             )
#                         results[function_name] = None
#                 else:
#                     if verbose:
#                         logger.warning(
#                             f"Function {function_name} not found."
#                         )
#                     results[function_name] = None

#         # Merge all results into a single string
#         merged_results = "\n".join(
#             f"{key}: {value}" for key, value in results.items()
#         )

#         return {"merged_results": merged_results}
#     except Exception as e:
#         logger.error(f"Error parsing and executing JSON: {e}")
#         return None


# def execute_and_log_function(
#     function_dict: Dict[str, Callable],
#     function_data: Dict[str, Any],
#     verbose: bool,
# ) -> Dict[str, Any]:
#     """
#     Executes a function from a given dictionary of functions and logs the execution details.

#     Args:
#         function_dict (Dict[str, Callable]): A dictionary containing the available functions.
#         function_data (Dict[str, Any]): A dictionary containing the function name and parameters.
#         verbose (bool): A flag indicating whether to log the execution details.

#     Returns:
#         Dict[str, Any]: A dictionary containing the function name and its result.

#     """
#     function_name = function_data.get("name")
#     parameters = function_data.get("parameters")

#     if verbose:
#         logger.info(
#             f"Executing function: {function_name} with parameters: {parameters}"
#         )

#     if function_name in function_dict:
#         try:
#             result = function_dict[function_name](**parameters)
#             if verbose:
#                 logger.info(
#                     f"Result for function {function_name}: {result}"
#                 )
#             return {function_name: str(result)}
#         except Exception as e:
#             if verbose:
#                 logger.error(
#                     f"Error executing function {function_name}: {e}"
#                 )
#             return {function_name: None}
#     else:
#         if verbose:
#             logger.warning(f"Function {function_name} not found.")
#         return {function_name: None}
