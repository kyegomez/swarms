import json
from typing import List, Any, Callable
import re

from swarms.utils.parse_code import extract_code_from_markdown
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="tool_parse_exec")


def parse_and_execute_json(
    functions: List[Callable[..., Any]],
    json_string: str,
    parse_md: bool = False,
    verbose: bool = False,
    max_retries: int = 3,
) -> str:
    """
    Parses and executes a JSON string containing function names and parameters.

    Args:
        functions (List[Callable[..., Any]]): A list of callable functions.
        json_string (str): The JSON string to parse and execute.
        parse_md (bool): Flag indicating whether to extract code from Markdown.
        verbose (bool): Flag indicating whether to enable verbose logging.
        return_str (bool): Flag indicating whether to return a JSON string.
        max_retries (int): Maximum number of retries for executing functions.

    Returns:
        dict: A dictionary containing the results of executing the functions with the parsed parameters.
    """
    if not functions or not json_string:
        raise ValueError("Functions and JSON string are required")

    if parse_md:
        try:
            code_blocks = re.findall(
                r"```(?:json)?\s*([\s\S]*?)```", json_string
            )
            if code_blocks and len(code_blocks) > 1:
                function_dict = {
                    func.__name__: func for func in functions
                }

                def process_json_block(json_block: str) -> dict:
                    try:
                        json_block = json_block.strip()
                        if not json_block:
                            raise ValueError("JSON block is empty")
                        data = json.loads(json_block)
                        function_list = []
                        if "functions" in data:
                            function_list = data["functions"]
                        elif "function" in data:
                            function_list = [data["function"]]
                        else:
                            function_list = [data]
                        if isinstance(function_list, dict):
                            function_list = [function_list]
                        function_list = [
                            f for f in function_list if f
                        ]

                        block_results = {}
                        for function_data in function_list:
                            function_name = function_data.get("name")
                            parameters = function_data.get(
                                "parameters", {}
                            )

                            if not function_name:
                                logger.warning(
                                    "Function data missing 'name' field"
                                )
                                continue

                            if function_name not in function_dict:
                                logger.warning(
                                    f"Function '{function_name}' not found"
                                )
                                block_results[function_name] = (
                                    "Error: Function not found"
                                )
                                continue

                            for attempt in range(max_retries):
                                try:
                                    result = function_dict[
                                        function_name
                                    ](**parameters)
                                    block_results[function_name] = (
                                        str(result)
                                    )
                                    logger.info(
                                        f"Result for {function_name}: {result}"
                                    )
                                    break
                                except Exception as e:
                                    logger.error(
                                        f"Attempt {attempt + 1} failed for {function_name}: {e}"
                                    )
                                    if attempt == max_retries - 1:
                                        block_results[
                                            function_name
                                        ] = f"Error after {max_retries} attempts: {str(e)}"
                        return block_results
                    except Exception as e:
                        logger.error(
                            f"Failed to process JSON block: {e}"
                        )
                        return {
                            "error": f"Failed to process block: {str(e)}"
                        }

                combined_results = {}
                for idx, block in enumerate(code_blocks, start=1):
                    combined_results[f"block_{idx}"] = (
                        process_json_block(block)
                    )
                return json.dumps(
                    {"results": combined_results}, indent=4
                )
            elif code_blocks:
                json_string = code_blocks[0]
            else:
                json_string = extract_code_from_markdown(json_string)
        except Exception as e:
            logger.error(
                f"Error extracting code blocks from Markdown: {e}"
            )
            return {"error": f"Markdown parsing failed: {str(e)}"}

    try:
        # Ensure JSON string is stripped of extraneous whitespace
        json_string = json_string.strip()
        if not json_string:
            raise ValueError(
                "JSON string is empty after stripping whitespace"
            )

        function_dict = {func.__name__: func for func in functions}

        if verbose:
            logger.info(
                f"Available functions: {list(function_dict.keys())}"
            )
            logger.info(f"Processing JSON: {json_string}")

        try:
            data = json.loads(json_string)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            return {"error": f"Invalid JSON format: {str(e)}"}

        function_list = []
        if "functions" in data:
            function_list = data["functions"]
        elif "function" in data:
            function_list = [data["function"]]
        else:
            function_list = [data]

        if isinstance(function_list, dict):
            function_list = [function_list]

        function_list = [f for f in function_list if f]

        if verbose:
            logger.info(f"Processing {len(function_list)} functions")

        results = {}

        for function_data in function_list:
            function_name = function_data.get("name")
            parameters = function_data.get("parameters", {})

            if not function_name:
                logger.warning("Function data missing 'name' field")
                continue

            if verbose:
                logger.info(
                    f"Executing {function_name} with parameters: {parameters}"
                )

            if function_name not in function_dict:
                logger.warning(
                    f"Function '{function_name}' not found"
                )
                results[function_name] = "Error: Function not found"
                continue

            for attempt in range(max_retries):
                try:
                    result = function_dict[function_name](
                        **parameters
                    )
                    results[function_name] = str(result)
                    if verbose:
                        logger.info(
                            f"Result for {function_name}: {result}"
                        )
                    break
                except Exception as e:
                    logger.error(
                        f"Attempt {attempt + 1} failed for {function_name}: {e}"
                    )
                    if attempt == max_retries - 1:
                        results[function_name] = (
                            f"Error after {max_retries} attempts: {str(e)}"
                        )

        data = {
            "results": results,
            "summary": "\n".join(
                f"{k}: {v}" for k, v in results.items()
            ),
        }

        return json.dumps(data, indent=4)

    except Exception as e:
        error = f"Unexpected error during execution: {str(e)}"
        logger.error(error)
        return {"error": error}
