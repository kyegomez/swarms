import json
import re
from typing import Any, List

from swarms.prompts.tools import SCENARIOS
from swarms.tools.base_tool import BaseTool
import inspect
from typing import Callable

from termcolor import colored


def scrape_tool_func_docs(fn: Callable) -> str:
    """
    Scrape the docstrings and parameters of a function decorated with `tool` and return a formatted string.

    Args:
        fn (Callable): The function to scrape.

    Returns:
        str: A string containing the function's name, documentation string, and a list of its parameters. Each parameter is represented as a line containing the parameter's name, default value, and annotation.
    """
    try:
        # If the function is a tool, get the original function
        if hasattr(fn, "func"):
            fn = fn.func

        signature = inspect.signature(fn)
        parameters = []
        for name, param in signature.parameters.items():
            parameters.append(
                f"Name: {name}, Type:"
                f" {param.default if param.default is not param.empty else 'None'},"
                " Annotation:"
                f" {param.annotation if param.annotation is not param.empty else 'None'}"
            )
        parameters_str = "\n".join(parameters)
        return (
            f"Function: {fn.__name__}\nDocstring:"
            f" {inspect.getdoc(fn)}\nParameters:\n{parameters_str}"
        )
    except Exception as error:
        print(
            colored(
                (
                    f"Error scraping tool function docs {error} try"
                    " optimizing your inputs with different"
                    " variables and attempt once more."
                ),
                "red",
            )
        )


def tool_find_by_name(tool_name: str, tools: List[Any]):
    """Find the tool by name"""
    for tool in tools:
        if tool.name == tool_name:
            return tool
    return None


def extract_tool_commands(text: str):
    """
    Extract the tool commands from the text

    Example:
    ```json
    {
        "tool": "tool_name",
        "params": {
            "tool1": "inputs",
            "param2": "value2"
        }
    }
    ```

    """
    # Regex to find JSON like strings
    pattern = r"```json(.+?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    json_commands = []
    for match in matches:
        try:
            json_commands = json.loads(match)
            json_commands.append(json_commands)
        except Exception as error:
            print(f"Error parsing JSON command: {error}")


def parse_and_execute_tools(response: str):
    """Parse and execute the tools"""
    json_commands = extract_tool_commands(response)
    for command in json_commands:
        tool_name = command.get("tool")
        params = command.get("parmas", {})
        execute_tools(tool_name, params)


def execute_tools(tool_name, params):
    """Execute the tool with the provided params"""
    tool = tool_find_by_name(tool_name)
    if tool:
        # Execute the tool with the provided parameters
        tool_result = tool.run(**params)
        print(tool_result)


def parse_tool_docs(tools: List[BaseTool]):
    """Parse the tool docs"""
    tool_docs = []
    for tool in tools:
        docs = tool_docs.append(scrape_tool_func_docs(tool))
    return str(docs)


def tools_prompt_prep(docs: str = None, scenarios: str = SCENARIOS):
    """
    Tools prompt prep

    Args:
        docs (str, optional): _description_. Defaults to None.
        scenarios (str, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    PROMPT = f"""
    # Task
    You will be provided with a list of APIs. These APIs will have a
    description and a list of parameters and return types for each tool. Your
    task involves creating varied, complex, and detailed user scenarios
    that require to call API calls. You must select what api to call based on 
    the context of the task and the scenario.

    For instance, given the APIs: SearchHotels, BookHotel, CancelBooking,
    GetNFLNews. Given that GetNFLNews is explicitly provided, your scenario
    should articulate something akin to:

    "The user wants to see if the Broncos won their last game (GetNFLNews).
    They then want to see if that qualifies them for the playoffs and who
    they will be playing against (GetNFLNews). The Broncos did make it into
    the playoffs, so the user wants watch the game in person. They want to
    look for hotels where the playoffs are occurring (GetNBANews +
    SearchHotels). After looking at the options, the user chooses to book a
    3-day stay at the cheapest 4-star option (BookHotel)."
    13

    This scenario exemplifies a scenario using 5 API calls. The scenario is
    complex, detailed, and concise as desired. The scenario also includes two
    APIs used in tandem, the required API, GetNBANews to search for the
    playoffs location and SearchHotels to find hotels based on the returned
    location. Usage of multiple APIs in tandem is highly desirable and will
    receive a higher score. Ideally each scenario should contain one or more
    instances of multiple APIs being used in tandem.

    Note that this scenario does not use all the APIs given and re-uses the "
    GetNBANews" API. Re-using APIs is allowed, but each scenario should
    involve as many different APIs as the user demands. Note that API usage is also included
    in the scenario, but exact parameters ar necessary. You must use a
    different combination of APIs for each scenario. All APIs must be used in
    at least one scenario. You can only use the APIs provided in the APIs
    section.
    
    Note that API calls are not explicitly mentioned and their uses are
    included in parentheses. This behaviour should be mimicked in your
    response.
    
    Output the tool usage in a strict json format with the function name and input to 
    the function. For example, Deliver your response in this format:
    
    ‘‘‘
    {scenarios}
    ‘‘‘
    # APIs
    ‘‘‘
    {docs}
    ‘‘‘
    # Response
    ‘‘‘
    """
    return PROMPT


def is_str_valid_func_output(
    output: str = None, function_map: callable = None
):
    """
    Check if the output is a valid JSON string, and if the function name in the JSON matches any name in the function map.

    Args:
        output (str): The output to check.
        function_map (dict): A dictionary mapping function names to functions.

    Returns:
        bool: True if the output is valid and the function name matches, False otherwise.
    """
    try:
        # Parse the output as JSON
        data = json.loads(output)

        # Check if the output matches the schema
        if (
            data.get("type") == "function"
            and "function" in data
            and "name" in data["function"]
        ):

            # Check if the function name matches any name in the function map
            function_name = data["function"]["name"]
            if function_name in function_map:
                return True

    except json.JSONDecodeError:
        pass

    return False
