"""Provider shapes accepted by ``tool_call_arguments``."""

from pydantic import BaseModel

from swarms.utils.str_to_dict import tool_call_arguments

ARGUMENTS = '{"confidence": 0.9, "estimated_cost": 5.0}'
PARSED = {"confidence": 0.9, "estimated_cost": 5.0}


class _Function(BaseModel):
    name: str
    arguments: str


class _ToolCall(BaseModel):
    function: _Function


def _object_call():
    return _ToolCall(
        function=_Function(name="bid", arguments=ARGUMENTS)
    )


def _dict_call():
    return {"function": {"name": "bid", "arguments": ARGUMENTS}}


def test_accepts_a_pydantic_tool_call_object():
    """Copies that only did .get("function") dropped object-shaped calls entirely."""
    assert tool_call_arguments([_object_call()]) == PARSED


def test_accepts_a_plain_dict_tool_call():
    """Copies that only did attribute access raised AttributeError on dicts."""
    assert tool_call_arguments([_dict_call()]) == PARSED


def test_accepts_the_repr_of_a_tool_call_list():
    """An agent whose output_type renders to text hands back the repr, not the list."""
    assert tool_call_arguments(str([_dict_call()])) == PARSED


def test_accepts_a_bare_call_outside_a_list():
    assert tool_call_arguments(_dict_call()) == PARSED
    assert tool_call_arguments(_object_call()) == PARSED


def test_returns_none_for_output_it_cannot_unwrap():
    """Callers supply their own default, so unusable output must come back as None."""
    unusable = [
        None,
        "",
        [],
        "not python at all",
        {"function": {"name": "bid", "arguments": "{not json"}},
        {"function": {"name": "bid", "arguments": "[1, 2]"}},
        {"no_function_key": 1},
    ]
    for value in unusable:
        assert tool_call_arguments(value) is None
