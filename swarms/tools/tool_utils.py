from typing import Any, List

import inspect
from typing import Callable
from swarms.utils.formatter import formatter


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
        (
            formatter.print_panel(
                f"Error scraping tool function docs {error} try"
                " optimizing your inputs with different"
                " variables and attempt once more."
            ),
        )

        raise error


def tool_find_by_name(tool_name: str, tools: List[Any]):
    """Find the tool by name"""
    for tool in tools:
        if tool.name == tool_name:
            return tool
    return None
