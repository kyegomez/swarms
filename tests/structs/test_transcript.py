"""
Tests for :class:`swarms.structs.transcript.Transcript` and its use on the
integer ``max_loops`` path.

Approach
--------
Fully offline. The ``Transcript`` tests are pure unit tests. The agent tests
build a real ``Agent`` (construction performs no network I/O) and monkeypatch
the single seam that would reach a provider, ``Agent.call_llm``, so the real
``_run`` control flow is exercised end to end.

The invariant under test throughout: an assistant message carrying
``tool_calls`` must be followed by one ``{"role": "tool", "tool_call_id": ...}``
message per call. A gap there makes the *next* request invalid, so it is the
property that matters most.

Run:
    cd /Users/swarms_wd/Desktop/research/swarms
    PYTHONPATH=. python3 -m pytest tests/structs/test_transcript.py -q -p no:randomly
"""

import json

import pytest

from swarms import Agent
from swarms.structs.transcript import Transcript


def tool_call(name, call_id="call_1", **arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def assert_calls_answered(messages):
    """Every assistant tool_calls turn is followed by its tool results."""
    for index, message in enumerate(messages):
        if message["role"] != "assistant" or not message.get(
            "tool_calls"
        ):
            continue
        expected = [c["id"] for c in message["tool_calls"]]
        following = [
            messages[j]["tool_call_id"]
            for j in range(
                index + 1,
                min(index + 1 + len(expected), len(messages)),
            )
            if messages[j]["role"] == "tool"
        ]
        assert following == expected, (
            f"assistant turn {index} left tool calls unanswered: "
            f"expected {expected}, got {following}"
        )


class TestTranscript:
    def test_records_a_tool_call_and_its_result(self):
        t = Transcript()
        t.append_user("read it")
        calls = t.record_assistant(
            [tool_call("read_file", file_path="a")]
        )

        assert len(calls) == 1 and calls[0]["name"] == "read_file"
        t.flush_tool_results(calls, {calls[0]["id"]: "contents"})

        assert [m["role"] for m in t.messages] == [
            "user",
            "assistant",
            "tool",
        ]
        assert t.messages[-1]["content"] == "contents"
        assert_calls_answered(t.messages)

    def test_text_only_response_is_an_assistant_turn(self):
        t = Transcript()
        calls = t.record_assistant("just talking")

        assert calls == []
        assert t.messages == [
            {"role": "assistant", "content": "just talking"}
        ]

    def test_missing_result_is_filled_not_skipped(self):
        """A gap would make the next request invalid, so it must be filled."""
        t = Transcript()
        calls = t.record_assistant([tool_call("grep", call_id="c9")])
        t.flush_tool_results(calls, {})

        assert t.messages[-1]["role"] == "tool"
        assert t.messages[-1]["tool_call_id"] == "c9"
        assert "no result recorded" in t.messages[-1]["content"]
        assert_calls_answered(t.messages)

    def test_multiple_calls_each_get_a_result(self):
        t = Transcript()
        calls = t.record_assistant(
            [
                tool_call("read_file", call_id="a"),
                tool_call("grep", call_id="b"),
            ]
        )
        t.flush_tool_results(calls, {"a": "one", "b": "two"})

        assert [m["role"] for m in t.messages] == [
            "assistant",
            "tool",
            "tool",
        ]
        assert_calls_answered(t.messages)

    def test_call_without_an_id_gets_a_synthesised_one(self):
        t = Transcript()
        calls = t.record_assistant(
            [{"function": {"name": "f", "arguments": "{}"}}]
        )
        assert calls and calls[0]["id"]
        t.flush_tool_results(calls, {})
        assert_calls_answered(t.messages)

    def test_batch_output_maps_one_value_per_call(self):
        t = Transcript()
        calls = [{"id": "a"}, {"id": "b"}]
        results = {}
        t.map_batch_results(calls, ["one", "two"], results)
        assert results == {"a": "one", "b": "two"}

    def test_combined_batch_output_is_recorded_against_every_call(
        self,
    ):
        t = Transcript()
        calls = [{"id": "a"}, {"id": "b"}]
        results = {}
        t.map_batch_results(calls, "combined", results)
        assert results == {"a": "combined", "b": "combined"}

    def test_messages_property_is_a_copy(self):
        t = Transcript()
        t.append_user("x")
        t.messages.append({"role": "user", "content": "mutated"})
        assert len(t) == 1


class TestIntegerMaxLoopsUsesMessages:
    """The fixed-loop path sends a real conversation, not a flattened string."""

    def _agent(self, **kwargs):
        return Agent(
            agent_name="IntPathTest",
            model_name="gpt-4o-mini",
            persistent_memory=False,
            print_on=False,
            verbose=False,
            autosave=False,
            # Keeps the suite offline: the summariser would otherwise make a
            # real provider call after each tool execution.
            tool_call_summary=False,
            **kwargs,
        )

    def test_call_llm_receives_messages_not_a_task_string(
        self, monkeypatch
    ):
        agent = self._agent(max_loops=1)
        seen = {}

        def capture(task=None, *args, **kwargs):
            seen["task"] = task
            seen["messages"] = kwargs.get("messages")
            return "done"

        monkeypatch.setattr(agent, "call_llm", capture)
        agent.run("hello")

        assert (
            seen["task"] is None
        ), "still sending a flattened task string"
        assert isinstance(seen["messages"], list)
        assert seen["messages"], "an empty conversation was sent"

    def test_the_task_appears_as_a_user_turn(self, monkeypatch):
        agent = self._agent(max_loops=1)
        seen = {}

        monkeypatch.setattr(
            agent,
            "call_llm",
            lambda task=None, *a, **k: (
                seen.setdefault("messages", k.get("messages")),
                "done",
            )[1],
        )
        agent.run("find the bug")

        users = [
            m["content"]
            for m in seen["messages"]
            if m["role"] == "user"
        ]
        assert any("find the bug" in u for u in users)

    def test_tool_results_are_paired_across_loops(self, monkeypatch):
        """The second loop must see the first loop's tool call answered."""

        def get_weather(city: str) -> str:
            """Return the weather for a city.

            Args:
                city: The city name.
            """
            return f"{city}: sunny"

        agent = self._agent(max_loops=2, tools=[get_weather])
        captured = []
        turn = {"n": 0}

        def scripted(task=None, *args, **kwargs):
            captured.append(kwargs.get("messages"))
            turn["n"] += 1
            if turn["n"] == 1:
                return [
                    tool_call(
                        "get_weather", call_id="w1", city="Paris"
                    )
                ]
            return "It is sunny in Paris."

        monkeypatch.setattr(agent, "call_llm", scripted)
        agent.run("weather in Paris?")

        assert len(captured) == 2, "the second loop did not run"
        second = captured[1]
        assert any(
            m["role"] == "tool" for m in second
        ), "the tool result was not sent as a tool message"
        assert_calls_answered(second)

    def test_transforms_keep_the_legacy_flattened_prompt(
        self, monkeypatch
    ):
        """`transforms` rewrites history into a string by design."""
        agent = self._agent(max_loops=1)
        agent.transforms = (
            object()
        )  # any non-None value selects that path
        seen = {}

        def capture(task=None, *args, **kwargs):
            seen["task"] = task
            seen["messages"] = kwargs.get("messages")
            return "done"

        monkeypatch.setattr(agent, "call_llm", capture)
        monkeypatch.setattr(
            "swarms.structs.agent.handle_transforms",
            lambda **kw: "FLATTENED",
        )
        agent.run("hello")

        assert seen["task"] == "FLATTENED"
        assert seen["messages"] is None


class TestTranscriptMirrorsIntoConversation:
    """Given a conversation, the transcript records the typed turns there
    too, so short_memory carries what the model saw."""

    def _conversation(self):
        from swarms.structs.conversation import Conversation

        return Conversation(time_enabled=False)

    def test_tool_calling_turn_and_results_are_recorded_typed(self):
        conv = self._conversation()
        t = Transcript(conversation=conv, agent_name="Agent")
        calls = t.record_assistant([tool_call("read_file", path="a")])
        t.flush_tool_results(calls, {calls[0]["id"]: "contents"})

        rows = conv.conversation_history
        assert rows[0]["role"] == "Agent"
        assert rows[0]["content"] is None
        assert (
            rows[0]["tool_calls"][0]["function"]["name"]
            == "read_file"
        )
        assert rows[1]["role"] == "tool"
        assert rows[1]["tool_call_id"] == calls[0]["id"]
        assert rows[1]["content"] == "contents"
        assert rows[1]["metadata"]["name"] == "read_file"

    def test_text_turn_is_recorded_once_under_the_agent_name(self):
        conv = self._conversation()
        t = Transcript(conversation=conv, agent_name="Agent")
        t.record_assistant("plain answer")

        assert [r["role"] for r in conv.conversation_history] == [
            "Agent"
        ]
        assert (
            conv.conversation_history[0]["content"] == "plain answer"
        )

    def test_placeholder_results_stay_out_of_the_conversation(self):
        """The placeholder keeps the request valid; nothing happened."""
        conv = self._conversation()
        t = Transcript(conversation=conv, agent_name="Agent")
        calls = t.record_assistant([tool_call("handoff")])
        t.flush_tool_results(calls, {})

        assert (
            t.messages[-1]["role"] == "tool"
        )  # request still well formed
        assert [r["role"] for r in conv.conversation_history] == [
            "Agent"
        ]

    def test_user_turns_are_left_to_the_caller(self):
        conv = self._conversation()
        t = Transcript(conversation=conv, agent_name="Agent")
        t.append_user("hello")

        assert conv.conversation_history == []

    def test_without_a_conversation_nothing_changes(self):
        t = Transcript()
        calls = t.record_assistant([tool_call("read_file", path="a")])
        t.flush_tool_results(calls, {calls[0]["id"]: "x"})
        assert len(t) == 2


def get_weather(city: str) -> str:
    """Report the weather for a city.

    Args:
        city: The city to report on.
    """
    return f"sunny in {city}"


class TestIntegerPathMemoryCarriesToolTurns(
    TestIntegerMaxLoopsUsesMessages
):
    """After a tool-using run, short_memory holds typed rows, and the next
    run rebuilds a well-formed request from them."""

    def test_memory_holds_the_typed_turns_after_a_run(
        self, monkeypatch
    ):
        agent = self._agent(max_loops=2, tools=[get_weather])
        scripted = iter(
            [[tool_call("get_weather", city="Paris")], "It is sunny."]
        )
        monkeypatch.setattr(
            agent, "call_llm", lambda *a, **k: next(scripted)
        )

        agent.run("Weather in Paris?")

        rows = agent.short_memory.conversation_history
        call_rows = [r for r in rows if r.get("tool_calls")]
        result_rows = [r for r in rows if r.get("tool_call_id")]
        assert len(call_rows) == 1
        assert call_rows[0]["role"] == agent.agent_name
        assert call_rows[0]["content"] is None
        assert len(result_rows) == 1
        assert (
            result_rows[0]["tool_call_id"]
            == call_rows[0]["tool_calls"][0]["id"]
        )

    def test_next_run_rebuilds_a_well_formed_request_from_memory(
        self, monkeypatch
    ):
        agent = self._agent(max_loops=2, tools=[get_weather])
        scripted = iter(
            [
                [tool_call("get_weather", city="Paris")],
                "It is sunny.",
                "Still sunny.",
            ]
        )
        seen = []

        def capture(*args, **kwargs):
            seen.append(kwargs.get("messages"))
            return next(scripted, "Still sunny.")

        monkeypatch.setattr(agent, "call_llm", capture)

        agent.run("Weather in Paris?")
        agent.run("And now?")

        second_run_request = seen[-1]
        assert_calls_answered(second_run_request)
        assistant_tool_turns = [
            m for m in second_run_request if m.get("tool_calls")
        ]
        assert len(assistant_tool_turns) == 1
        assert assistant_tool_turns[0]["content"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-q", "-p", "no:randomly"])
