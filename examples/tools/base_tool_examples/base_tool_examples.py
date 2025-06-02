from swarms.tools.base_tool import (
    BaseTool,
    ToolValidationError,
    ToolExecutionError,
    ToolNotFoundError,
)
import json


def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a location.

    Args:
        location (str): The city or location to get weather for
        unit (str, optional): Temperature unit ('celsius' or 'fahrenheit'). Defaults to 'celsius'.

    Returns:
        str: A string describing the current weather at the location

    Examples:
        >>> get_current_weather("New York")
        'Weather in New York is likely sunny and 75° Celsius'
        >>> get_current_weather("London", "fahrenheit")
        'Weather in London is likely sunny and 75° Fahrenheit'
    """
    return f"Weather in {location} is likely sunny and 75° {unit.title()}"


def add_numbers(a: int, b: int) -> int:
    """Add two numbers together.

    Args:
        a (int): First number to add
        b (int): Second number to add

    Returns:
        int: The sum of a and b

    Examples:
        >>> add_numbers(2, 3)
        5
        >>> add_numbers(-1, 1)
        0
    """
    return a + b


# Example with improved error handling and logging
try:
    # Create BaseTool instance with verbose logging
    tool_manager = BaseTool(
        verbose=True,
        auto_execute_tool=False,
    )

    print(
        json.dumps(
            tool_manager.func_to_dict(get_current_weather),
            indent=4,
        )
    )

    print(
        json.dumps(
            tool_manager.multiple_functions_to_dict(
                [get_current_weather, add_numbers]
            ),
            indent=4,
        )
    )

except (
    ToolValidationError,
    ToolExecutionError,
    ToolNotFoundError,
) as e:
    print(f"Tool error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
