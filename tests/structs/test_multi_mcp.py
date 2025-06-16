import asyncio
import json
from swarms.structs.agent import Agent, extract_json_from_response
from swarms.structs.agent import execute_mcp_call
from unittest.mock import patch


def test_handle_multiple_mcp_tools():
    agent = Agent(agent_name="Test", llm=None, max_loops=1)
    urls = ["http://server1", "http://server2"]
    payloads = [
        {
            "function_name": "tool1",
            "server_url": "http://server1",
            "payload": {"a": 1},
        },
        {
            "function_name": "tool2",
            "server_url": "http://server2",
            "payload": {},
        },
    ]
    called = []

    async def fake_exec(
        function_name, server_url, payload, *args, **kwargs
    ):
        called.append((function_name, server_url, payload))
        return "ok"

    with patch(
        "swarms.structs.agent.execute_mcp_call", side_effect=fake_exec
    ):
        agent.handle_multiple_mcp_tools(urls, json.dumps(payloads))

    assert called == [
        ("tool1", "http://server1", {"a": 1}),
        ("tool2", "http://server2", {}),
    ]


def test_extract_json_from_response():
    payloads = [
        {"function_name": "foo", "server_url": "http://x", "payload": {"x": 1}}
    ]
    text = "Random text" + json.dumps(payloads) + " end"
    result = extract_json_from_response(text)
    assert result == payloads
