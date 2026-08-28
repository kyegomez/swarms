"""Context delivery for the helpers in ``swarms.structs.swarming_architectures``."""

import asyncio

import pytest

from swarms.structs.agent import Agent
from swarms.structs.swarming_architectures import (
    broadcast,
    circular_swarm,
    star_swarm,
)


class RecordingAgent(Agent):
    """An Agent that answers locally and records what it was handed."""

    def __init__(self, name, calls):
        self.agent_name = name
        self._calls = calls

    def run(self, task=None, messages=None, *args, **kwargs):
        self._calls.append(
            {
                "agent": self.agent_name,
                "task": task,
                "messages": messages or [],
            }
        )
        return f"{self.agent_name}-answer"


def _agents(names, calls):
    return [RecordingAgent(name, calls) for name in names]


def _contents(call):
    return [m["content"] for m in call["messages"]]


def test_circular_swarm_delivers_typed_turns_not_one_blob():
    """Each speaker is its own labelled turn, not a flattened string."""
    calls = []
    circular_swarm(agents=_agents(["A", "B"], calls), tasks=["do it"])

    first, second = calls[0], calls[1]

    assert first["agent"] == "A"
    assert first["task"] == "do it"
    assert first["messages"] == []

    assert second["agent"] == "B"
    # split_last_turn hands the newest turn over as the task, so A's answer
    # is either the task or one of the prior turns - never inside a blob.
    seen = _contents(second) + [second["task"]]
    assert any(
        "A-answer" in str(c) for c in seen
    ), f"B never saw A's answer: {second}"
    assert "do it" in seen
    assert all(
        isinstance(m, dict) and m["role"] in ("user", "assistant")
        for m in second["messages"]
    )


def test_an_agent_sees_its_own_prior_output_as_assistant():
    """Two tasks means A speaks twice; the second time it reads itself back."""
    calls = []
    circular_swarm(
        agents=_agents(["A", "B"], calls), tasks=["one", "two"]
    )

    second_turn_for_a = [c for c in calls if c["agent"] == "A"][1]
    own = [
        m
        for m in second_turn_for_a["messages"]
        if m["content"] == "A-answer"
    ]

    assert len(own) == 1, f"expected one own turn: {own}"
    assert own[0]["role"] == "assistant"


def test_the_task_is_recorded_once_per_task_not_once_per_agent():
    """The shared turn is added for the task, not re-added for every agent."""
    calls = []
    circular_swarm(
        agents=_agents(["A", "B", "C"], calls), tasks=["do it"]
    )

    last = calls[-1]
    assert (
        _contents(last).count("do it") + [last["task"]].count("do it")
        == 1
    ), f"the task appears more than once: {last}"


def test_star_swarm_centre_receives_typed_turns():
    calls = []
    star_swarm(agents=_agents(["Centre", "X"], calls), tasks=["go"])

    centre = calls[0]
    assert centre["agent"] == "Centre"
    assert centre["task"] == "go"
    assert isinstance(centre["messages"], list)


def test_broadcast_agents_see_the_sender_as_a_labelled_turn():
    calls = []
    sender = RecordingAgent("Sender", calls)
    asyncio.run(
        broadcast(
            sender=sender,
            agents=_agents(["R1", "R2"], calls),
            task="announce",
        )
    )

    receiver = [c for c in calls if c["agent"] == "R1"][0]
    seen = _contents(receiver) + [receiver["task"]]

    assert any(
        "Sender-answer" in str(c) for c in seen
    ), f"R1 did not see the sender's turn: {receiver}"


def test_nothing_is_handed_a_flattened_role_colon_blob():
    """The old form rendered the whole room as 'Role: content' prose."""
    calls = []
    circular_swarm(
        agents=_agents(["A", "B"], calls), tasks=["one", "two"]
    )

    for call in calls:
        assert "User: " not in str(
            call["task"]
        ), f"{call['agent']} received flattened prose: {call['task']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
