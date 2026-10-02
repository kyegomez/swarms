import pytest

from swarms.structs.one_to_three import OneToThree, one_to_three


class EchoAgent:
    def __init__(self, name):
        self.agent_name = name
        self.calls = []

    def run(self, task=None, messages=None, **kwargs):
        self.calls.append({"task": task, "messages": messages or []})
        return f"{self.agent_name} got: {task}"


def three():
    return [EchoAgent("R1"), EchoAgent("R2"), EchoAgent("R3")]


def test_one_to_three_sender_then_three_receivers():
    result = one_to_three(
        EchoAgent("Sender"), three(), "task", output_type="dict"
    )

    assert [m["role"] for m in result] == [
        "User",
        "Sender",
        "R1",
        "R2",
        "R3",
    ]
    assert result[1]["content"] == "Sender got: task"


@pytest.mark.parametrize("count", [0, 2, 4])
def test_one_to_three_requires_exactly_three(count):
    receivers = [EchoAgent(f"R{i}") for i in range(count)]

    with pytest.raises(ValueError):
        one_to_three(EchoAgent("Sender"), receivers, "task")

    with pytest.raises(ValueError):
        OneToThree(EchoAgent("Sender"), receivers)


def test_one_to_three_rejects_empty_task():
    with pytest.raises(ValueError):
        one_to_three(EchoAgent("Sender"), three(), "")


def test_one_to_three_class_matches_function():
    swarm = OneToThree(
        EchoAgent("Sender"), three(), output_type="dict"
    )

    result = swarm.run("task")

    assert [m["role"] for m in result] == [
        "User",
        "Sender",
        "R1",
        "R2",
        "R3",
    ]
    assert swarm.name == "OneToThree"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
