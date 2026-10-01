"""
Pin the requests both agent loops send to those sent before #2386.

Two requests changed on purpose and have their own tests: reasoning loop
markers are no longer sent, and the end-of-plan summary sends the typed
conversation.
"""

import copy
import itertools
import json

import pytest

from swarms import Agent

LIMIT = 80


def tc(name, call_id, **arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def get_weather(city: str) -> str:
    """Return the weather for a city.

    Args:
        city: The city name.
    """
    return f"{city}: sunny"


def get_time(city: str) -> str:
    """Return the local time for a city.

    Args:
        city: The city name.
    """
    return f"{city}: noon"


class _FakeSummaryLLM:
    def run(self, *args, **kwargs):
        return "SUMMARY OF TOOL OUTPUT"


def build_agent(**overrides):
    kwargs = {
        "agent_name": "EquivAgent",
        "model_name": "gpt-4o-mini",
        "persistent_memory": False,
        "print_on": False,
        "verbose": False,
        "autosave": False,
        "tool_call_summary": False,
        "context_compression": False,
    }
    kwargs.update(overrides)
    agent = Agent(**kwargs)
    agent.temp_llm_instance_for_tool_summary = (
        lambda: _FakeSummaryLLM()
    )
    return agent


def script(agent, responses):
    """
    Script the model's replies and return the requests it receives.

    Args:
        agent (Agent): The agent to script.
        responses (list): The replies, in order.

    Returns:
        list: One dict per request, with its task and messages.
    """
    queue, requests = list(responses), []

    def fake_call_llm(task=None, *args, **kwargs):
        requests.append(
            {
                "task": task,
                "messages": copy.deepcopy(kwargs.get("messages")),
            }
        )
        return queue.pop(0) if queue else "no further action"

    agent.call_llm = fake_call_llm
    return requests


def plan(*step_ids):
    return [
        tc(
            "create_plan",
            "p1",
            task_description="test task",
            steps=[
                {
                    "step_id": step_id,
                    "description": f"do {step_id}",
                    "priority": "high",
                    "dependencies": [],
                }
                for step_id in step_ids
            ],
        )
    ]


EARLIER = [
    {"role": "user", "content": "earlier question"},
    {"role": "assistant", "content": "earlier answer"},
]
WEATHER_TURNS = [
    [tc("get_weather", "w1", city="Paris")],
    "It is sunny in Paris.",
]
AUTO_TURNS = [
    plan("step1"),
    [tc("get_weather", "w1", city="Paris")],
    [
        tc(
            "subtask_done",
            "d1",
            task_id="step1",
            summary="got weather",
            success=True,
        ),
        tc(
            "complete_task",
            "c1",
            task_id="main",
            summary="all done",
            success=True,
        ),
    ],
    "FINAL SUMMARY",
]


def _run(agent, responses, *tasks, **run_kwargs):
    requests = script(agent, copy.deepcopy(responses))
    for task in tasks:
        agent.run(task, **run_kwargs)
    return requests


SCENARIOS = {
    "int_text": lambda: _run(
        build_agent(max_loops=1), ["done"], "hello"
    ),
    "int_tool": lambda: _run(
        build_agent(
            max_loops=2,
            reasoning_prompt_on=False,
            tools=[get_weather],
        ),
        WEATHER_TURNS,
        "weather in Paris?",
    ),
    "int_tool_summary": lambda: _run(
        build_agent(
            max_loops=2,
            reasoning_prompt_on=False,
            tools=[get_weather],
            tool_call_summary=True,
        ),
        WEATHER_TURNS,
        "weather in Paris?",
    ),
    "int_messages": lambda: _run(
        build_agent(
            max_loops=2,
            reasoning_prompt_on=False,
            tools=[get_weather],
        ),
        WEATHER_TURNS,
        "and now?",
        messages=EARLIER,
    ),
    "int_two_calls": lambda: _run(
        build_agent(
            max_loops=2,
            reasoning_prompt_on=False,
            tools=[get_weather, get_time],
        ),
        [
            [
                tc("get_weather", "w1", city="Paris"),
                tc("get_time", "t1", city="Paris"),
            ],
            "Sunny at noon.",
        ],
        "weather and time in Paris?",
    ),
    "int_second_run": lambda: _run(
        build_agent(max_loops=1),
        ["first answer", "second answer"],
        "first question",
        "second question",
    ),
    "auto_complete_task": lambda: _run(
        build_agent(max_loops="auto", tools=[get_weather]),
        AUTO_TURNS,
        "check the weather",
    ),
    "auto_messages": lambda: _run(
        build_agent(max_loops="auto", tools=[get_weather]),
        AUTO_TURNS,
        "check the weather",
        messages=EARLIER,
    ),
    "auto_think": lambda: _run(
        build_agent(max_loops="auto", tools=[get_weather]),
        [
            plan("step1"),
            [
                tc(
                    "think",
                    "k1",
                    analysis="need weather",
                    next_actions=["call tool"],
                    confidence=0.9,
                )
            ],
            [tc("get_weather", "w1", city="Paris")],
            [
                tc(
                    "subtask_done",
                    "d1",
                    task_id="step1",
                    summary="ok",
                    success=True,
                ),
                tc(
                    "complete_task",
                    "c1",
                    task_id="main",
                    summary="done",
                    success=True,
                ),
            ],
            "FINAL SUMMARY",
        ],
        "check the weather",
    ),
}


def normalise(message):
    """
    Return a message in the compact form the expected requests use.

    Args:
        message (dict): A chat-completions message.

    Returns:
        tuple: The role, then the content or the call details.
    """

    def short(content):
        if isinstance(content, str) and len(content) > LIMIT:
            return content[:LIMIT] + "…"
        return content

    if message.get("tool_calls"):
        return (
            "assistant",
            [
                (
                    call["function"]["name"],
                    call["id"],
                    call["function"]["arguments"],
                )
                for call in message["tool_calls"]
            ],
        )
    if message["role"] == "tool":
        return (
            "tool",
            message["tool_call_id"],
            short(message["content"]),
        )
    return (message["role"], short(message["content"]))


# Captured from master before the migration: the length of every request, and the last one.
EXPECTED = {
    "int_text": (
        [1],
        [
            ("user", "hello"),
        ],
    ),
    "int_tool": (
        [2, 4],
        [
            (
                "assistant",
                "[{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Return…",
            ),
            ("user", "weather in Paris?"),
            (
                "assistant",
                [("get_weather", "w1", '{"city": "Paris"}')],
            ),
            (
                "tool",
                "w1",
                "Function 'get_weather' result:\nParis: sunny",
            ),
        ],
    ),
    "int_tool_summary": (
        [2, 4],
        [
            (
                "assistant",
                "[{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Return…",
            ),
            ("user", "weather in Paris?"),
            (
                "assistant",
                [("get_weather", "w1", '{"city": "Paris"}')],
            ),
            (
                "tool",
                "w1",
                "Function 'get_weather' result:\nParis: sunny",
            ),
        ],
    ),
    "int_messages": (
        [3, 5],
        [
            ("user", "earlier question"),
            ("assistant", "earlier answer"),
            ("user", "and now?"),
            (
                "assistant",
                [("get_weather", "w1", '{"city": "Paris"}')],
            ),
            (
                "tool",
                "w1",
                "Function 'get_weather' result:\nParis: sunny",
            ),
        ],
    ),
    "int_two_calls": (
        [2, 5],
        [
            (
                "assistant",
                "[{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Return…",
            ),
            ("user", "weather and time in Paris?"),
            (
                "assistant",
                [
                    ("get_weather", "w1", '{"city": "Paris"}'),
                    ("get_time", "t1", '{"city": "Paris"}'),
                ],
            ),
            (
                "tool",
                "w1",
                "Function 'get_weather' result:\nParis: sunny",
            ),
            (
                "tool",
                "t1",
                "Function 'get_time' result:\nParis: noon",
            ),
        ],
    ),
    "int_second_run": (
        [1, 3],
        [
            ("user", "first question"),
            ("assistant", "first answer"),
            ("user", "second question"),
        ],
    ),
    "auto_complete_task": (
        [2, 5, 7, 11],
        [
            ("user", "check the weather"),
            (
                "user",
                "You need to create a comprehensive plan for the following task:\n\ncheck the weath…",
            ),
            (
                "assistant",
                [
                    (
                        "create_plan",
                        "p1",
                        '{"task_description": "test task", "steps": [{"step_id": "step1", "description": "do step1", "priority": "high", "dependencies": []}]}',
                    )
                ],
            ),
            (
                "tool",
                "p1",
                "Plan created successfully with 1 subtasks",
            ),
            (
                "user",
                "You are currently working on subtask: step1\nDescription: do step1\n\nCurrent statu…",
            ),
            (
                "assistant",
                [("get_weather", "w1", '{"city": "Paris"}')],
            ),
            (
                "tool",
                "w1",
                "Function 'get_weather' result:\nParis: sunny",
            ),
            (
                "assistant",
                [
                    (
                        "subtask_done",
                        "d1",
                        '{"task_id": "step1", "summary": "got weather", "success": true}',
                    ),
                    (
                        "complete_task",
                        "c1",
                        '{"task_id": "main", "summary": "all done", "success": true}',
                    ),
                ],
            ),
            ("tool", "d1", "Subtask step1 marked as completed"),
            (
                "tool",
                "c1",
                "Task Completion Summary\n\nTask ID: main\nStatus: Success\nSummary: all done\n\nSubtas…",
            ),
            (
                "user",
                "All subtasks are complete.\nGenerate a clear, comprehensive final summary using t…",
            ),
        ],
    ),
    "auto_messages": (
        [4, 7, 9, 13],
        [
            ("user", "earlier question"),
            ("assistant", "earlier answer"),
            ("user", "check the weather"),
            (
                "user",
                "You need to create a comprehensive plan for the following task:\n\ncheck the weath…",
            ),
            (
                "assistant",
                [
                    (
                        "create_plan",
                        "p1",
                        '{"task_description": "test task", "steps": [{"step_id": "step1", "description": "do step1", "priority": "high", "dependencies": []}]}',
                    )
                ],
            ),
            (
                "tool",
                "p1",
                "Plan created successfully with 1 subtasks",
            ),
            (
                "user",
                "You are currently working on subtask: step1\nDescription: do step1\n\nCurrent statu…",
            ),
            (
                "assistant",
                [("get_weather", "w1", '{"city": "Paris"}')],
            ),
            (
                "tool",
                "w1",
                "Function 'get_weather' result:\nParis: sunny",
            ),
            (
                "assistant",
                [
                    (
                        "subtask_done",
                        "d1",
                        '{"task_id": "step1", "summary": "got weather", "success": true}',
                    ),
                    (
                        "complete_task",
                        "c1",
                        '{"task_id": "main", "summary": "all done", "success": true}',
                    ),
                ],
            ),
            ("tool", "d1", "Subtask step1 marked as completed"),
            (
                "tool",
                "c1",
                "Task Completion Summary\n\nTask ID: main\nStatus: Success\nSummary: all done\n\nSubtas…",
            ),
            (
                "user",
                "All subtasks are complete.\nGenerate a clear, comprehensive final summary using t…",
            ),
        ],
    ),
    "auto_think": (
        [2, 5, 7, 9, 13],
        [
            ("user", "check the weather"),
            (
                "user",
                "You need to create a comprehensive plan for the following task:\n\ncheck the weath…",
            ),
            (
                "assistant",
                [
                    (
                        "create_plan",
                        "p1",
                        '{"task_description": "test task", "steps": [{"step_id": "step1", "description": "do step1", "priority": "high", "dependencies": []}]}',
                    )
                ],
            ),
            (
                "tool",
                "p1",
                "Plan created successfully with 1 subtasks",
            ),
            (
                "user",
                "You are currently working on subtask: step1\nDescription: do step1\n\nCurrent statu…",
            ),
            (
                "assistant",
                [
                    (
                        "think",
                        "k1",
                        '{"analysis": "need weather", "next_actions": ["call tool"], "confidence": 0.9}',
                    )
                ],
            ),
            (
                "tool",
                "k1",
                "ERROR: think failed with TypeError: AutonomousAgentLoop._think_tool() missing 1 …",
            ),
            (
                "assistant",
                [("get_weather", "w1", '{"city": "Paris"}')],
            ),
            (
                "tool",
                "w1",
                "Function 'get_weather' result:\nParis: sunny",
            ),
            (
                "assistant",
                [
                    (
                        "subtask_done",
                        "d1",
                        '{"task_id": "step1", "summary": "ok", "success": true}',
                    ),
                    (
                        "complete_task",
                        "c1",
                        '{"task_id": "main", "summary": "done", "success": true}',
                    ),
                ],
            ),
            ("tool", "d1", "Subtask step1 marked as completed"),
            (
                "tool",
                "c1",
                "Task Completion Summary\n\nTask ID: main\nStatus: Success\nSummary: done\n\nSubtask Br…",
            ),
            (
                "user",
                "All subtasks are complete.\nGenerate a clear, comprehensive final summary using t…",
            ),
        ],
    ),
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_requests_match_the_transcript_era(name):
    lengths, last = EXPECTED[name]
    requests = SCENARIOS[name]()

    assert all(r["task"] is None for r in requests)
    bodies = [r["messages"] for r in requests]
    assert [len(body) for body in bodies] == lengths
    assert [normalise(m) for m in bodies[-1]] == last


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_request_extends_the_one_before(name):
    """A stable prefix is what lets the provider cache the request."""
    bodies = [r["messages"] for r in SCENARIOS[name]()]
    for earlier, later in itertools.pairwise(bodies):
        assert later[: len(earlier)] == earlier


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_tool_call_is_answered_in_every_request(name):
    for request in SCENARIOS[name]():
        body = request["messages"]
        for index, message in enumerate(body):
            if not message.get("tool_calls"):
                continue
            ids = [call["id"] for call in message["tool_calls"]]
            following = body[index + 1 : index + 1 + len(ids)]
            assert [m.get("tool_call_id") for m in following] == ids


def test_reasoning_loop_markers_are_not_sent():
    """Changed on purpose: only loop 1's marker used to leak in, as a trailing assistant turn."""
    agent = build_agent(max_loops=2, tools=[get_weather])
    requests = _run(agent, WEATHER_TURNS, "weather in Paris?")

    sent = [m for r in requests for m in r["messages"]]
    assert not any(
        "Internal Reasoning Loop" in str(m.get("content"))
        for m in sent
    )
    assert "Current Internal Reasoning Loop: 1/2" in (
        agent.short_memory.return_history_as_string()
    )
    assert requests[0]["messages"][-1] == {
        "role": "user",
        "content": "weather in Paris?",
    }


def test_the_end_of_plan_summary_sends_the_typed_conversation():
    """Changed on purpose: it used to send the history as one flattened string."""
    agent = build_agent(max_loops="auto", tools=[get_weather])
    requests = _run(
        agent,
        [
            plan("step1"),
            [
                tc(
                    "subtask_done",
                    "d1",
                    task_id="step1",
                    summary="ok",
                    success=True,
                )
            ],
            "FINAL SUMMARY",
        ],
        "check the weather",
    )

    summary_request = requests[-1]
    assert summary_request["task"] is None
    body = summary_request["messages"]
    previous = requests[-2]["messages"]
    assert body[: len(previous)] == previous

    tail = body[len(previous) :]
    assert [m["role"] for m in tail] == ["assistant", "tool", "user"]
    assert tail[0]["tool_calls"][0]["id"] == "d1"
    assert tail[1]["tool_call_id"] == "d1"
    assert tail[2]["content"].startswith("All subtasks are complete.")


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
