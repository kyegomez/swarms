import pytest
from swarms.structs.agent import Agent


def get_weather(city: str) -> str:
    """Get the weather for a city.

    Args:
        city (str): The city name.
    """
    return f"sunny in {city}"


def get_time(city: str) -> str:
    """Get the time for a city.

    Args:
        city (str): The city name.
    """
    return f"noon in {city}"


def test_add_tool_to_agent_without_initial_tools():
    """Agent constructed without tools should allow add_tool without AttributeError."""
    agent = Agent(
        agent_name="b",
        model_name="gpt-4o-mini",
        print_on=False,
    )
    assert agent.tools is None or agent.tools == []

    agent.add_tool(get_time)

    assert agent.tools == [get_time]
    assert "get_time" in agent.list_tools()
    assert agent.tool_struct is not None
    assert agent.tool_struct.tools == [get_time]
    assert agent.llm is not None
    assert agent.llm.tools_list_dictionary is not None
    llm_tool_names = [
        s["function"]["name"]
        for s in agent.llm.tools_list_dictionary
        if isinstance(s, dict) and "function" in s
    ]
    assert "get_time" in llm_tool_names


def test_add_and_remove_tool_updates_all_registries():
    """add_tool and remove_tool must update tool_struct, llm, and list_tools."""
    agent = Agent(
        agent_name="a",
        model_name="gpt-4o-mini",
        tools=[get_weather],
        print_on=False,
    )

    # Add tool
    agent.add_tool(get_time)
    assert [t.__name__ for t in agent.tools] == ["get_weather", "get_time"]
    assert "get_weather" in agent.list_tools()
    assert "get_time" in agent.list_tools()

    llm_tool_names = [
        s["function"]["name"]
        for s in agent.llm.tools_list_dictionary
        if isinstance(s, dict) and "function" in s
    ]
    assert "get_weather" in llm_tool_names
    assert "get_time" in llm_tool_names
    assert [t.__name__ for t in agent.tool_struct.tools] == [
        "get_weather",
        "get_time",
    ]

    # Remove tool
    agent.remove_tool(get_weather)
    assert [t.__name__ for t in agent.tools] == ["get_time"]
    assert agent.list_tools() == ["get_time"]

    llm_tool_names_after = [
        s["function"]["name"]
        for s in agent.llm.tools_list_dictionary
        if isinstance(s, dict) and "function" in s
    ]
    assert "get_weather" not in llm_tool_names_after
    assert "get_time" in llm_tool_names_after
    assert [t.__name__ for t in agent.tool_struct.tools] == ["get_time"]

    # Verify execution via tool_struct works for added tool
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "get_time", "arguments": '{"city": "Paris"}'},
    }
    result = agent.tool_struct.execute_function_calls_from_api_response([call])
    assert any("noon in Paris" in str(r) for r in result)


def test_add_tools_and_remove_tools_bulk():
    """Bulk adding and removing tools updates all registries."""
    agent = Agent(
        agent_name="c",
        model_name="gpt-4o-mini",
        print_on=False,
    )

    agent.add_tools([get_weather, get_time])
    assert set(agent.list_tools()) >= {"get_weather", "get_time"}
    assert [t.__name__ for t in agent.tool_struct.tools] == [
        "get_weather",
        "get_time",
    ]

    agent.remove_tools([get_weather])
    assert "get_weather" not in agent.list_tools()
    assert "get_time" in agent.list_tools()
    assert [t.__name__ for t in agent.tool_struct.tools] == ["get_time"]
