import asyncio

import pytest

from swarms.structs.broadcast import Broadcast, broadcast


class EchoAgent:
    def __init__(self, name):
        self.agent_name = name
        self.calls = []

    def run(self, task=None, messages=None, **kwargs):
        self.calls.append({"task": task, "messages": messages or []})
        return f"{self.agent_name} got: {task}"


def test_broadcast_sender_then_every_receiver():
    sender = EchoAgent("Sender")
    receivers = [EchoAgent("R1"), EchoAgent("R2"), EchoAgent("R3")]

    result = asyncio.run(
        broadcast(sender, receivers, "news", output_type="dict")
    )

    assert [m["role"] for m in result] == [
        "User",
        "Sender",
        "R1",
        "R2",
        "R3",
    ]
    # Every receiver sees the sender's turn, as the task or as a prior turn.
    for receiver in receivers:
        seen = receiver.calls[0]["task"] + str(
            receiver.calls[0]["messages"]
        )
        assert "Sender got: news" in seen


def test_broadcast_accepts_nested_receiver_lists():
    sender = EchoAgent("Sender")
    nested = [[EchoAgent("R1"), EchoAgent("R2")], [EchoAgent("R3")]]

    result = asyncio.run(
        broadcast(sender, nested, "news", output_type="dict")
    )

    assert [m["role"] for m in result][2:] == ["R1", "R2", "R3"]


def test_broadcast_rejects_empty_receivers():
    with pytest.raises(ValueError):
        asyncio.run(broadcast(EchoAgent("S"), [], "news"))


def test_broadcast_class_runs_synchronously():
    group = Broadcast(
        EchoAgent("Sender"),
        [EchoAgent("R1"), EchoAgent("R2")],
        output_type="dict",
    )

    result = group.run("news")

    assert [m["role"] for m in result] == [
        "User",
        "Sender",
        "R1",
        "R2",
    ]
    assert group.name == "Broadcast"


def test_broadcast_class_flattens_receivers_on_init():
    group = Broadcast(
        EchoAgent("Sender"),
        [[EchoAgent("R1")], [EchoAgent("R2")]],
    )

    assert [a.agent_name for a in group.receivers] == ["R1", "R2"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
