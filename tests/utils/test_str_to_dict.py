from types import SimpleNamespace

from swarms.utils.str_to_dict import tool_call_arguments

ARGUMENTS = '{"confidence": 0.9, "estimated_cost": 5.0}'
PARSED = {"confidence": 0.9, "estimated_cost": 5.0}


def _dict_call():
    return {"function": {"name": "bid", "arguments": ARGUMENTS}}


def _attribute_call():
    return SimpleNamespace(
        function=SimpleNamespace(name="bid", arguments=ARGUMENTS)
    )


class _DumpableCall:

    def model_dump(self):
        return _dict_call()


def test_accepts_a_plain_dict_tool_call():
    assert tool_call_arguments([_dict_call()]) == PARSED


def test_accepts_an_object_whose_function_is_an_attribute():
    assert tool_call_arguments([_attribute_call()]) == PARSED


def test_accepts_an_object_that_serialises_itself():
    assert tool_call_arguments([_DumpableCall()]) == PARSED


def test_accepts_the_repr_of_a_tool_call_list():
    assert tool_call_arguments(str([_dict_call()])) == PARSED


def test_accepts_a_bare_call_outside_a_list():
    assert tool_call_arguments(_dict_call()) == PARSED
    assert tool_call_arguments(_attribute_call()) == PARSED


def test_returns_none_for_output_it_cannot_unwrap():
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


def test_decodes_bytes_arguments():
    call = {
        "function": {"name": "bid", "arguments": ARGUMENTS.encode()}
    }
    assert tool_call_arguments(call) == PARSED
