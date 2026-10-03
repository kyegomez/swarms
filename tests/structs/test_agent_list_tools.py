from unittest.mock import AsyncMock, patch

import pytest

from swarms import Agent
from swarms.structs.autonomous_loop_utils import (
    get_autonomous_planning_tools,
)
from swarms.tools.mcp_manager import MCPManager


def get_weather(city: str) -> str:
    """
    Get the weather for a city.

    Args:
        city (str): The city name.

    Returns:
        str: The weather.
    """
    return f"sunny in {city}"


def get_time(zone: str) -> str:
    """
    Get the time in a time zone.

    Args:
        zone (str): The time zone.

    Returns:
        str: The time.
    """
    return f"noon in {zone}"


def _schema(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


MCP_TOOLS = [_schema("lookup_stock"), _schema("get_news")]


def _agent(**kwargs):
    kwargs.setdefault("max_loops", 1)
    kwargs.setdefault("agent_name", "lister")
    with patch("swarms.agents.tool_manager.LiteLLM"), patch(
        "swarms.agents.llm_manager.LiteLLM"
    ):
        return Agent(
            model_name="gpt-5.4",
            print_on=False,
            verbose=False,
            persistent_memory=False,
            **kwargs,
        )


@pytest.fixture
def mcp_fetch():
    """Stands in for one MCP server connection; counts connections."""
    with patch.object(
        MCPManager,
        "_alist_tools",
        new=AsyncMock(return_value=MCP_TOOLS),
    ) as fetch:
        yield fetch


def test_tools_only():
    agent = _agent(tools=[get_weather, get_time])
    assert agent.list_tools() == ["get_weather", "get_time"]


def test_tools_list_dictionary_only():
    agent = _agent(tools_list_dictionary=[_schema("raw_tool")])
    assert agent.list_tools() == ["raw_tool"]


def test_mcp_only(mcp_fetch):
    agent = _agent(mcp_url="http://localhost:8000/mcp")
    assert agent.list_tools() == ["lookup_stock", "get_news"]


def test_tools_and_mcp_in_order(mcp_fetch):
    agent = _agent(
        tools=[get_weather], mcp_url="http://localhost:8000/mcp"
    )
    assert agent.list_tools() == [
        "get_weather",
        "lookup_stock",
        "get_news",
    ]


def test_handoffs_add_handoff_task():
    researcher = _agent(agent_name="researcher")
    agent = _agent(tools=[get_weather], handoffs=[researcher])
    assert agent.list_tools() == ["get_weather", "handoff_task"]


def test_dynamic_tools_include_deferred_names(mcp_fetch):
    agent = _agent(
        tools=[get_weather, get_time],
        mcp_url="http://localhost:8000/mcp",
        dynamic_tools=True,
    )
    names = agent.list_tools()

    assert set(agent.tool_loader.deferred_names) >= {
        "get_weather",
        "get_time",
    }
    assert names == [
        "get_time",
        "get_weather",
        "lookup_stock",
        "get_news",
        "tool_search",
    ]


def test_auto_loop_tools_included():
    agent = _agent(tools=[get_weather], max_loops="auto")
    expected = [
        schema["function"]["name"]
        for schema in get_autonomous_planning_tools()
        if schema["function"]["name"] != "think"
    ]
    assert agent.list_tools() == ["get_weather"] + expected


def test_auto_loop_think_tool_and_selected_tools():
    agent = _agent(
        max_loops="auto",
        think_tool=True,
        selected_tools=["create_plan", "think", "complete_task"],
    )
    assert agent.list_tools() == [
        "create_plan",
        "think",
        "complete_task",
    ]


def test_no_duplicates():
    agent = _agent(
        tools=[get_weather],
        tools_list_dictionary=[_schema("get_weather")],
    )
    assert agent.list_tools() == ["get_weather"]


def test_unreachable_mcp_returns_other_names():
    agent = _agent(
        tools=[get_weather],
        mcp_url="http://localhost:8000/mcp",
        llm=object(),
    )
    with patch.object(
        MCPManager,
        "_alist_tools",
        new=AsyncMock(side_effect=ConnectionError("refused")),
    ):
        assert agent.list_tools() == ["get_weather"]


def test_list_tools_twice_opens_one_connection(mcp_fetch):
    agent = _agent(mcp_url="http://localhost:8000/mcp", llm=object())
    assert mcp_fetch.await_count == 0

    agent.list_tools()
    agent.list_tools()

    assert mcp_fetch.await_count == 1


def test_reuses_schemas_fetched_at_construction(mcp_fetch):
    agent = _agent(mcp_url="http://localhost:8000/mcp")
    fetched = mcp_fetch.await_count

    agent.list_tools()

    assert fetched == 1
    assert mcp_fetch.await_count == 1
