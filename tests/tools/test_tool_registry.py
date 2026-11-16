from swarms.tools.base_tool import BaseTool
from swarms.tools.tool_registry import ToolStorage, tool_registry

storage = ToolStorage(name="tools", description="Registered tools")


@tool_registry(storage)
def get_weather(city: str, unit: str = "celsius") -> str:
    """Return the weather for a city."""
    return f"20 {unit} in {city}"


@tool_registry(storage)
def get_time(city: str) -> str:
    """Return the local time in a city."""
    return f"12:00 in {city}"


def test_registered_tools_keep_their_name_docstring_and_signature():
    weather, time = BaseTool().multiple_functions_to_dict(
        [get_weather, get_time]
    )

    assert weather["function"]["name"] == "get_weather"
    assert (
        weather["function"]["description"]
        == "Return the weather for a city."
    )
    assert weather["function"]["parameters"]["required"] == ["city"]
    assert time["function"]["name"] == "get_time"
    assert get_weather("Paris") == "20 celsius in Paris"
