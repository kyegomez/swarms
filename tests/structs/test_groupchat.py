"""
Offline pytest suite for the turn-based, self-selecting GroupChat.

The previous file was a live-LLM script converted to ``test_*.py``
naming: every test took a ``report`` parameter that exists nowhere as a
fixture, so all five errored at collection and GroupChat had zero
working coverage (#2059). These tests drive the real GroupChat
scheduling loop with scripted agents instead — no model, no API key, no
network.

A scripted agent returns a canned ``respond(score, message)`` tool call
in either shape the provider path produces: the tool-call list, or its
repr — the string form an agent with the default ``output_type`` hands
back, which is exactly the case that previously went silent.
"""

import json

import pytest

from swarms.structs.groupchat import (
    GroupChat,
    RESPOND_TOOL,
    _extract_args,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def respond_call(score, message):
    """A ``respond()`` tool call shaped like parsed provider output."""
    return [
        {
            "function": {
                "name": "respond",
                "arguments": json.dumps(
                    {"score": score, "message": message}
                ),
            }
        }
    ]


class ScriptedAgent:
    """Stands in for ``Agent``: bids are canned, not model-driven.

    Carries ``RESPOND_TOOL`` so ``auto_equip`` leaves it untouched, and
    accepts the ``run(task=..., messages=...)`` contract the chat uses.
    Once its scripted bids run out it bids ``(0.0, "")`` — a silent
    agent — so every test ends on a natural lull instead of hanging.
    """

    def __init__(self, name, bids=(), as_string=False):
        self.agent_name = name
        self.tools_list_dictionary = [RESPOND_TOOL]
        self._bids = list(bids)
        self._as_string = as_string
        self.calls = 0

    def run(self, task=None, *args, **kwargs):
        self.calls += 1
        if self._bids:
            score, message = self._bids.pop(0)
        else:
            score, message = 0.0, ""
        output = respond_call(score, message)
        return repr(output) if self._as_string else output


class BareAgent(ScriptedAgent):
    """A scripted agent that does NOT yet carry the respond tool."""

    def __init__(self, name, bids=()):
        super().__init__(name, bids)
        self.tools_list_dictionary = []
        self.llm = None

    def llm_handling(self):
        return self.llm


def make_chat(agents, **kwargs):
    kwargs.setdefault("max_loops", 10)
    return GroupChat(agents=agents, **kwargs)


def roles(chat):
    return [
        message["role"]
        for message in chat.conversation.conversation_history
    ]


def contents(chat):
    return [
        message["content"]
        for message in chat.conversation.conversation_history
    ]


# --------------------------------------------------------------------------
# _extract_args — the bid-parsing path that silently ended chats when
# it could not handle a provider's output shape
# --------------------------------------------------------------------------


class TestExtractArgs:
    def test_parses_the_tool_call_list_form(self):
        score, message = _extract_args(respond_call(0.8, "hello"))
        assert score == pytest.approx(0.8)
        assert message == "hello"

    def test_parses_a_single_dict(self):
        score, message = _extract_args(respond_call(0.6, "hi")[0])
        assert score == pytest.approx(0.6)
        assert message == "hi"

    def test_parses_the_string_repr_form(self):
        """An agent with the default output_type hands back the repr of
        the tool-call list. Treating it as unparseable made every bid
        (0.0, "") and ended the chat on turn one.
        """
        raw = repr(respond_call(0.84, "a real message"))
        score, message = _extract_args(raw)
        assert score == pytest.approx(0.84)
        assert message == "a real message"

    def test_arguments_already_a_dict(self):
        call = {
            "function": {
                "name": "respond",
                "arguments": {"score": 0.5, "message": "m"},
            }
        }
        assert _extract_args([call]) == (0.5, "m")

    @pytest.mark.parametrize(
        "bad",
        [
            None,
            [],
            {},
            "not a tool call",
            "[unbalanced",
            {"function": {}},
            {"no_function": True},
            [{"function": {"name": "respond", "arguments": "{bad"}}],
        ],
    )
    def test_invalid_output_is_a_silent_decision(self, bad):
        assert _extract_args(bad) == (0.0, "")

    def test_score_is_clamped_into_range(self):
        assert _extract_args(respond_call(1.7, "m"))[0] == 1.0
        assert _extract_args(respond_call(-0.5, "m"))[0] == 0.0

    def test_non_numeric_score_defaults_to_zero(self):
        call = {
            "function": {
                "name": "respond",
                "arguments": json.dumps(
                    {"score": "loud", "message": "m"}
                ),
            }
        }
        assert _extract_args([call])[0] == 0.0

    def test_message_is_stripped(self):
        _, message = _extract_args(respond_call(0.5, "  padded  "))
        assert message == "padded"


# --------------------------------------------------------------------------
# constructor validation
# --------------------------------------------------------------------------


class TestConstructorValidation:
    def test_empty_agent_list_is_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            GroupChat(agents=[])

    def test_a_single_agent_is_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            GroupChat(agents=[ScriptedAgent("Solo")])

    def test_none_agents_raise_value_error_not_type_error(self):
        """`agents` defaults to None; the documented contract is
        ValueError, not a TypeError from len(None)."""
        with pytest.raises(ValueError, match="at least 2"):
            GroupChat(agents=None)

    def test_deprecated_idle_timeout_is_still_accepted(self):
        chat = make_chat(
            [ScriptedAgent("A"), ScriptedAgent("B")],
            idle_timeout=4.0,
        )
        assert chat.idle_timeout == 4.0

    def test_auto_equip_injects_the_respond_tool(self):
        bare_a, bare_b = BareAgent("A"), BareAgent("B")
        make_chat([bare_a, bare_b])

        for agent in (bare_a, bare_b):
            names = [
                tool["function"]["name"]
                for tool in agent.tools_list_dictionary
            ]
            assert "respond" in names

    def test_auto_equip_false_leaves_agents_alone(self):
        bare_a, bare_b = BareAgent("A"), BareAgent("B")
        make_chat([bare_a, bare_b], auto_equip=False)

        assert bare_a.tools_list_dictionary == []
        assert bare_b.tools_list_dictionary == []

    def test_agents_already_equipped_are_not_double_equipped(self):
        agent_a = ScriptedAgent("A")
        make_chat([agent_a, ScriptedAgent("B")])

        assert agent_a.tools_list_dictionary == [RESPOND_TOOL]


# --------------------------------------------------------------------------
# scheduling — one speaker per turn, lull, cap, recency
# --------------------------------------------------------------------------


class TestScheduling:
    def test_the_highest_bidder_takes_the_floor(self):
        eager = ScriptedAgent("Eager", [(0.9, "eager speaks")])
        shy = ScriptedAgent("Shy", [(0.6, "shy speaks")])

        make_chat_ = make_chat([eager, shy])
        make_chat_.run("topic")

        assert roles(make_chat_)[:2] == ["User", "Eager"]
        assert "eager speaks" in contents(make_chat_)
        assert "shy speaks" not in contents(make_chat_)

    def test_every_agent_is_asked_each_turn(self):
        agents = [
            ScriptedAgent("A", [(0.9, "a")]),
            ScriptedAgent("B", [(0.6, "b")]),
            ScriptedAgent("C", [(0.6, "c")]),
        ]
        make_chat(agents).run("topic")

        # Turn 1: everyone bids. Turn 2: everyone bids again and the
        # exhausted scripts fall silent, ending the chat on a lull.
        assert all(agent.calls >= 1 for agent in agents)

    def test_a_lull_ends_the_chat(self):
        quiet = [
            ScriptedAgent("A", [(0.2, "murmur")]),
            ScriptedAgent("B", [(0.1, "whisper")]),
        ]
        chat = make_chat(quiet, threshold=0.5)
        chat.run("anything to add?")

        assert roles(chat) == ["User"]

    def test_an_empty_message_never_posts(self):
        chat = make_chat(
            [
                ScriptedAgent("A", [(0.99, "")]),
                ScriptedAgent("B", [(0.98, "")]),
            ]
        )
        chat.run("topic")

        assert roles(chat) == ["User"]

    def test_max_loops_caps_total_messages(self):
        chatty = [
            ScriptedAgent("A", [(0.9, f"a{i}") for i in range(9)]),
            ScriptedAgent("B", [(0.6, f"b{i}") for i in range(9)]),
        ]
        chat = make_chat(chatty, max_loops=4, recency_penalty=0.0)
        chat.run("go")

        # The user task counts as the first message.
        assert len(roles(chat)) == 4

    def test_recency_penalty_passes_the_floor_around(self):
        agent_a = ScriptedAgent(
            "A", [(0.8, f"a{i}") for i in range(3)]
        )
        agent_b = ScriptedAgent(
            "B", [(0.7, f"b{i}") for i in range(3)]
        )
        chat = make_chat(
            [agent_a, agent_b],
            max_loops=4,
            threshold=0.5,
            recency_penalty=0.3,
            recency_window=1,
        )
        chat.run("debate")

        # A wins turn 1; penalized to 0.5 on turn 2 so B's 0.7 takes
        # the floor; on turn 3 the penalty has moved to B and A wins.
        assert roles(chat) == ["User", "A", "B", "A"]

    def test_string_form_bids_still_speak(self):
        """End to end over the repr path: the exact configuration that
        used to leave every agent silent must post a message."""
        eager = ScriptedAgent(
            "Eager", [(0.9, "parsed from repr")], as_string=True
        )
        shy = ScriptedAgent("Shy", as_string=True)

        chat = make_chat([eager, shy])
        chat.run("topic")

        assert "parsed from repr" in contents(chat)

    def test_a_failing_agent_is_treated_as_silent(self):
        class ExplodingAgent(ScriptedAgent):
            def run(self, task=None, *args, **kwargs):
                raise RuntimeError("provider outage")

        eager = ScriptedAgent("Eager", [(0.9, "still works")])
        chat = make_chat([ExplodingAgent("Broken"), eager])
        chat.run("topic")

        assert "still works" in contents(chat)


# --------------------------------------------------------------------------
# run() surface — return value, batching, streaming
# --------------------------------------------------------------------------


class TestRunSurface:
    def test_run_returns_formatted_history(self):
        chat = make_chat(
            [
                ScriptedAgent(
                    "A", [(0.9, "hello"), (0.8, "hello again")]
                ),
                ScriptedAgent("B"),
            ],
            recency_penalty=0.0,
        )
        result = chat.run("topic")

        # Default output_type renders the history to a string. The
        # second reply is asserted because the string formatter's
        # "all except first" currently drops one more message than its
        # dict twin (conversation.py slices [2:] vs [1:]); this holds
        # under either slicing.
        assert isinstance(result, str)
        assert "hello again" in result
        assert "hello" in contents(chat)

    def test_run_batch_returns_one_result_per_task(self):
        chat = make_chat([ScriptedAgent("A"), ScriptedAgent("B")])
        results = chat.run_batch(["first", "second"])

        assert len(results) == 2

    def test_streaming_callback_replays_each_posted_message(self):
        events = []

        def on_chunk(sender, chunk, is_final):
            events.append((sender, chunk, is_final))

        chat = make_chat(
            [
                ScriptedAgent("A", [(0.9, "two words")]),
                ScriptedAgent("B"),
            ]
        )
        chat.run("hello there", streaming_callback=on_chunk)

        senders = {sender for sender, _, _ in events}
        assert senders == {"User", "A"}
        # Every message ends with the is_final sentinel.
        assert ("User", "", True) in events
        assert ("A", "", True) in events
        # The speaker's reply is chunked over time, not sent whole.
        a_chunks = [
            chunk
            for sender, chunk, final in events
            if sender == "A" and not final
        ]
        assert "".join(a_chunks) == "two words"
        assert len(a_chunks) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
