"""Tests for ``swarms.tools.tool_registry.ToolStorage``.

No model and no network: the registry only holds callables.
"""

from swarms.tools.tool_registry import ToolStorage


def greet(name: str) -> str:
    """Greet someone."""
    return f"hello {name}"


def farewell(name: str) -> str:
    """Say goodbye."""
    return f"bye {name}"


def test_builds_without_a_name_or_description():
    storage = ToolStorage()

    assert storage.list_tools() == []


def test_the_tools_passed_in_are_registered():
    storage = ToolStorage(name="demo", tools=[greet, farewell])

    assert storage.get_tool("greet") is greet
    assert sorted(storage.list_tools()) == ["farewell", "greet"]
