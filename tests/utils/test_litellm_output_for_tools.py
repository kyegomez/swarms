"""Tests for ``LiteLLM.output_for_tools``.

Regression coverage for https://github.com/kyegomez/swarms/issues/2424:
in MCP mode the wrapper rebuilt every tool call as a bare
``{"function": {...}}`` dict, dropping the call's ``id`` and ``type``.
Local tools then never ran (the executor did not recognise the call)
and tool results could not be keyed back to their call.
"""

import json
from types import SimpleNamespace

from swarms.tools.base_tool import BaseTool
from swarms.utils.litellm_wrapper import LiteLLM


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _response_with(*tool_calls):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(tool_calls=list(tool_calls))
            )
        ]
    )


def test_mcp_single_call_keeps_id_and_type():
    response = _response_with(
        _tool_call("call_1", "get_weather", '{"location": "Paris"}')
    )
    out = LiteLLM(mcp_call=True).output_for_tools(response)
    assert out == {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": '{"location": "Paris"}',
        },
    }


def test_mcp_single_call_without_id_keeps_legacy_shape():
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="ping", arguments="{}")
    )
    out = LiteLLM(mcp_call=True).output_for_tools(
        _response_with(tool_call)
    )
    assert out == {"function": {"name": "ping", "arguments": "{}"}}


def test_mcp_multiple_calls_keep_ids_and_types():
    response = _response_with(
        _tool_call("call_1", "get_weather", '{"location": "Paris"}'),
        _tool_call("call_2", "get_time", '{"tz": "UTC"}'),
    )
    out = LiteLLM(mcp_call=True).output_for_tools(response)
    assert [call["id"] for call in out] == ["call_1", "call_2"]
    assert all(call["type"] == "function" for call in out)
    assert [call["function"]["name"] for call in out] == [
        "get_weather",
        "get_time",
    ]


def test_mcp_output_executes_local_tool():
    """The exact failure from the issue: a local tool in a mixed
    MCP setup must run, and its result must be returned."""
    ran = []

    def write_note(text: str):
        """Record a note."""
        ran.append(text)
        return f"noted: {text}"

    response = _response_with(
        _tool_call("call_9", "write_note", '{"text": "hi"}')
    )
    out = LiteLLM(mcp_call=True).output_for_tools(response)
    results = BaseTool(
        tools=[write_note]
    ).execute_function_calls_from_api_response(out)
    assert ran == ["hi"]
    assert results and "noted: hi" in results[0]


def test_standard_path_still_returns_raw_tool_calls():
    tool_call = _tool_call("call_1", "get_weather", "{}")
    out = LiteLLM(mcp_call=False).output_for_tools(
        _response_with(tool_call)
    )
    assert out == [tool_call]
