import pytest

from swarms.structs.one_to_one import OneToOne, one_to_one


class EchoAgent:
    """Answers with its name and the task it was handed; records what it saw."""

    def __init__(self, name):
        self.agent_name = name
        self.calls = []

    def run(self, task=None, messages=None, **kwargs):
        self.calls.append({"task": task, "messages": messages or []})
        return f"{self.agent_name} got: {task}"


def test_one_to_one_records_user_sender_receiver():
    sender, receiver = EchoAgent("Sender"), EchoAgent("Receiver")

    result = one_to_one(sender, receiver, "hello", output_type="dict")

    assert [m["role"] for m in result] == [
        "User",
        "Sender",
        "Receiver",
    ]
    assert result[1]["content"] == "Sender got: hello"
    # The receiver answers the sender's turn, not the original task.
    assert "Sender got: hello" in receiver.calls[0]["task"]


def test_one_to_one_loops_alternate():
    sender, receiver = EchoAgent("Sender"), EchoAgent("Receiver")

    result = one_to_one(
        sender, receiver, "hello", max_loops=2, output_type="dict"
    )

    assert [m["role"] for m in result] == [
        "User",
        "Sender",
        "Receiver",
        "Sender",
        "Receiver",
    ]
    # On the second loop the sender is answering the receiver's reply.
    seen = sender.calls[1]["task"] + str(sender.calls[1]["messages"])
    assert "Receiver got:" in seen


def test_one_to_one_rejects_empty_task():
    with pytest.raises(ValueError):
        one_to_one(EchoAgent("A"), EchoAgent("B"), "")


def test_one_to_one_class_matches_function():
    pair = OneToOne(
        EchoAgent("Sender"), EchoAgent("Receiver"), output_type="dict"
    )

    result = pair.run("hello")

    assert [m["role"] for m in result] == [
        "User",
        "Sender",
        "Receiver",
    ]
    assert pair.name == "OneToOne"


def test_one_to_one_class_reusable_across_tasks():
    pair = OneToOne(
        EchoAgent("Sender"), EchoAgent("Receiver"), output_type="dict"
    )

    first = pair.run("one")
    second = pair.run("two")

    assert first[0]["content"] == "one"
    assert second[0]["content"] == "two"
    assert len(second) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
