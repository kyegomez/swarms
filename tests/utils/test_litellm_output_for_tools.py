"""Tests for ``LiteLLM.output_for_tools``.

Regression coverage for https://github.com/kyegomez/swarms/issues/2424:
in MCP mode the wrapper rebuilt every tool call as a bare
``{"function": {...}}`` dict, dropping the call's ``id`` and ``type``.
Local tools then never ran (the executor did not recognise the call)
and tool results could not be keyed back to their call.
"""

from types import SimpleNamespace

import pytest

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


@pytest.mark.parametrize(
    "call_ids, expected_ids",
    [
        pytest.param(["call_1"], ["call_1"], id="single"),
        pytest.param(
            ["call_1", "call_2"], ["call_1", "call_2"], id="multiple"
        ),
        pytest.param([None], [None], id="no-id"),
    ],
)
def test_mcp_calls_keep_provider_fields(call_ids, expected_ids):
    tool_calls = [
        (
            _tool_call(call_id, "ping", "{}")
            if call_id is not None
            else SimpleNamespace(
                function=SimpleNamespace(name="ping", arguments="{}")
            )
        )
        for call_id in call_ids
    ]
    out = LiteLLM(mcp_call=True).output_for_tools(
        _response_with(*tool_calls)
    )
    expected = []
    for call_id in expected_ids:
        call = {"function": {"name": "ping", "arguments": "{}"}}
        if call_id is not None:
            call.update(id=call_id, type="function")
        expected.append(call)
    assert out == (expected[0] if len(expected) == 1 else expected)


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
