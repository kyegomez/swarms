import pytest

from swarms.tools.tool_choice import forced_tool_choice

SCHEMA = {"type": "function", "function": {"name": "respond"}}
FORCED = {"type": "function", "function": {"name": "respond"}}


@pytest.mark.parametrize(
    "tool, reasoning, expected",
    [
        (SCHEMA, {}, FORCED),
        ("respond", {}, FORCED),
        (SCHEMA, {"reasoning_effort": "low"}, "auto"),
        (SCHEMA, {"reasoning_enabled": True}, "auto"),
        (SCHEMA, {"reasoning_enabled": False}, FORCED),
    ],
)
def test_forced_tool_choice_names_the_tool_unless_reasoning_is_on(
    tool, reasoning, expected
):
    assert forced_tool_choice(tool, **reasoning) == expected
