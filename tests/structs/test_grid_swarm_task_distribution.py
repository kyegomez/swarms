"""``grid_swarm`` must run every task it is handed, and leave the caller's list alone."""

import pytest

from swarms.structs.swarming_architectures import grid_swarm


class RecordingAgent:
    """An agent that answers locally and records the task it was handed."""

    def __init__(self, name, calls):
        self.agent_name = name
        self._calls = calls

    def run(self, task, *args, **kwargs):
        self._calls.append((self.agent_name, task))
        return f"{self.agent_name}-answer"


def test_grid_swarm_runs_every_task_once_and_keeps_the_caller_list():
    """A 5-agent roster is not a 2x2 grid: the uneven tail must still be worked."""
    calls = []
    agents = [RecordingAgent(f"A{i}", calls) for i in range(5)]
    tasks = [f"t{i}" for i in range(5)]

    result = grid_swarm(agents=agents, tasks=tasks)

    assert [task for _, task in calls] == tasks
    assert tasks == [
        f"t{i}" for i in range(5)
    ], "grid_swarm consumed the caller's task list"
    turns = [
        row["content"] for row in result if row["role"] != "User"
    ]
    assert turns == [f"{name}-answer" for name, _ in calls]
    assert len(turns) == len(tasks)


def test_grid_swarm_cycles_the_roster_in_order():
    """Iteration ``i`` goes to ``agents[i % len(agents)]``, wrapping as needed."""
    calls = []
    agents = [RecordingAgent(f"A{i}", calls) for i in range(3)]

    grid_swarm(agents=agents, tasks=[f"t{i}" for i in range(5)])

    assert calls == [
        ("A0", "t0"),
        ("A1", "t1"),
        ("A2", "t2"),
        ("A0", "t3"),
        ("A1", "t4"),
    ]


@pytest.mark.parametrize(
    "agents,tasks",
    [([], ["t0"]), ([RecordingAgent("A0", [])], [])],
)
def test_grid_swarm_rejects_empty_inputs(agents, tasks):
    """The documented ValueError, not an empty conversation or an IndexError."""
    with pytest.raises(ValueError):
        grid_swarm(agents=agents, tasks=tasks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
