"""Tests for :mod:`swarms.structs.context_utils` — typed delivery of a shared
conversation, including tool turns."""

import json

from swarms.structs.context_utils import messages_for
from swarms.structs.conversation import Conversation


def _call(name="read_file", call_id="call_1", **arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def _conversation():
    return Conversation(time_enabled=False)


class TestMessagesForToolTurns:
    def test_the_recipients_own_tool_use_is_delivered_typed(self):
        conv = _conversation()
        conv.add("User", "read a.txt")
        conv.add("Reader", None, tool_calls=[_call(path="a.txt")])
        conv.add(
            "tool",
            "contents",
            tool_call_id="call_1",
            metadata={"name": "read_file"},
        )
        conv.add("Reader", "It says contents.")

        messages = messages_for("Reader", conv)

        assert messages[0] == {
            "role": "user",
            "content": "read a.txt",
        }
        assert messages[1]["role"] == "assistant"
        assert messages[1]["tool_calls"][0]["id"] == "call_1"
        assert messages[2] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "contents",
        }
        assert messages[3] == {
            "role": "assistant",
            "content": "It says contents.",
        }

    def test_a_peers_tool_use_is_described_in_prose(self):
        """Only the caller may answer its own tool calls, so another
        agent sees what happened as text."""
        conv = _conversation()
        conv.add("Reader", None, tool_calls=[_call(path="a.txt")])
        conv.add(
            "tool",
            "contents",
            tool_call_id="call_1",
            metadata={"name": "read_file"},
        )

        messages = messages_for("Writer", conv)

        assert [m["role"] for m in messages] == ["user", "user"]
        assert messages[0]["content"].startswith(
            "Reader → read_file("
        )
        assert messages[1]["content"] == "read_file → contents"
        assert not any("tool_calls" in m for m in messages)

    def test_an_orphan_result_is_prose_not_a_tool_message(self):
        """A tool message the request cannot pair would be rejected."""
        conv = _conversation()
        conv.add(
            "tool",
            "contents",
            tool_call_id="call_9",
            metadata={"name": "read_file"},
        )

        messages = messages_for("Reader", conv)

        assert messages == [
            {"role": "user", "content": "read_file → contents"}
        ]

    def test_a_result_lands_right_after_its_call_despite_log_rows(
        self,
    ):
        """The loop logs a 'Tool Executor' row before the typed result
        is written; the request must still pair call and result."""
        conv = _conversation()
        conv.add("Reader", None, tool_calls=[_call(path="a.txt")])
        conv.add("Tool Executor", "read_file ran")
        conv.add(
            "tool",
            "contents",
            tool_call_id="call_1",
            metadata={"name": "read_file"},
        )

        messages = messages_for("Reader", conv)

        assert [m["role"] for m in messages] == [
            "assistant",
            "tool",
            "user",
        ]
        assert messages[1]["tool_call_id"] == "call_1"

    def test_a_text_turn_closes_the_open_calls(self):
        conv = _conversation()
        conv.add("Reader", None, tool_calls=[_call(path="a.txt")])
        conv.add("Reader", "never mind")
        conv.add(
            "tool",
            "late",
            tool_call_id="call_1",
            metadata={"name": "read_file"},
        )

        messages = messages_for("Reader", conv)

        assert messages[-1] == {
            "role": "user",
            "content": "read_file → late",
        }

    def test_plain_conversations_are_unchanged(self):
        conv = _conversation()
        conv.add("User", "draft it")
        conv.add("Writer", "here is a draft")
        conv.add("Editor", "needs a stronger opening")

        assert messages_for("Writer", conv) == [
            {"role": "user", "content": "draft it"},
            {"role": "assistant", "content": "here is a draft"},
            {
                "role": "user",
                "content": "Editor: needs a stronger opening",
            },
        ]
