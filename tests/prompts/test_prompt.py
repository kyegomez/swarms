import json

from swarms.prompts import Prompt


def get_weather(city: str) -> str:
    """
    Get the weather for a city.

    Args:
        city (str): The city name.

    Returns:
        str: The weather.
    """
    return "sunny"


def test_add_tools_appends_schema_as_a_new_version():
    prompt = Prompt(content="You are a weather assistant.")

    content = prompt.add_tools([get_weather])

    base, schema = content.split("\n\n", 1)
    assert base == "You are a weather assistant."
    assert json.loads(schema)["functions"][0]["name"] == "get_weather"
    assert prompt.edit_history[-1] == content
    prompt.rollback(0)
    assert prompt.content == "You are a weather assistant."
