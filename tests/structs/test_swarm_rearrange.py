import pytest

from swarms.structs.swarm_rearrange import (
    SwarmRearrange,
    swarm_arrange,
)


class RecordingSwarm:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def run(self, task, img=None):
        self.calls.append((self.name, task))
        return f"{self.name} out"


def test_a_swarm_can_run_in_more_than_one_step():
    calls = []
    swarms = [RecordingSwarm("A", calls), RecordingSwarm("B", calls)]

    result = SwarmRearrange(swarms=swarms, flow="A -> B -> A").run(
        "task"
    )

    assert [name for name, _ in calls] == ["A", "B", "A"]
    assert calls[0][1] == "task"
    assert "A out" in calls[1][1]
    assert "B out" in calls[2][1]
    assert "A out" in result

    calls.clear()
    swarm_arrange(swarms=swarms, flow="A -> B -> A", task="task")

    assert [name for name, _ in calls] == ["A", "B", "A"]


def test_two_human_checkpoints_call_the_callback_twice():
    calls = []
    seen = []

    def approve(task):
        seen.append(task)
        return "approved"

    flow = SwarmRearrange(
        swarms=[
            RecordingSwarm("A", calls),
            RecordingSwarm("B", calls),
        ],
        flow="A -> H -> B -> H",
        human_in_the_loop=True,
        custom_human_in_the_loop=approve,
    )

    assert flow.run("task") == "approved"
    assert len(seen) == 2
    assert [name for name, _ in calls] == ["A", "B"]


def test_a_swarm_repeated_within_one_step_raises():
    calls = []
    flow = SwarmRearrange(
        swarms=[
            RecordingSwarm("A", calls),
            RecordingSwarm("B", calls),
        ],
        flow="A -> B, B",
    )

    with pytest.raises(ValueError, match="one step"):
        flow.run("task")
    assert calls == []
