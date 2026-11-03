import pytest

from swarms.tools import ToolStorage


def greet(name: str) -> str:
    """Greet someone."""
    return f"hello {name}"


def test_duplicate_tool_is_not_logged():
    storage = ToolStorage(name="demo", description="demo")
    storage.add_tool(greet)
    with pytest.raises(ValueError):
        storage.add_tool(greet)

    assert [t.name for t in storage.tool_storage_schema.tools] == [
        "greet"
    ]
