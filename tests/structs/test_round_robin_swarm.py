import pytest
from swarms.structs.round_robin import RoundRobinSwarm
from swarms.structs.agent import Agent


@pytest.fixture
def round_robin_swarm():
    agents = [Agent(name=f"Agent{i}") for i in range(3)]
    return RoundRobinSwarm(agents=agents, verbose=True, max_loops=2)


def test_init(round_robin_swarm):
    assert isinstance(round_robin_swarm, RoundRobinSwarm)
    assert round_robin_swarm.verbose is True
    assert round_robin_swarm.max_loops == 2
    assert len(round_robin_swarm.agents) == 3


def test_run(round_robin_swarm):
    task = "test_task"
    result = round_robin_swarm.run(task)
    assert result == task
    assert round_robin_swarm.index == 0


def test_prior_turns_are_typed_not_flattened_into_the_task():
    """Each agent receives the shared history as chat turns, not one user blob."""

    class _Memory:
        def __init__(self):
            self.last = ""

        def get_final_message_content(self):
            return self.last

    class _RecordingAgent:
        def __init__(self, name):
            self.agent_name = name
            self.short_memory = _Memory()
            self.calls = []

        def run(self, task=None, messages=None, **kwargs):
            self.calls.append(
                {"task": str(task), "messages": messages}
            )
            self.short_memory.last = f"{self.agent_name}-out"
            return f"{self.agent_name}-out"

    first = _RecordingAgent("A")
    second = _RecordingAgent("B")

    RoundRobinSwarm(agents=[first, second], max_loops=2).run(
        "original task"
    )

    last = second.calls[-1]
    assert (
        "A-out" not in last["task"]
    ), f"transcript flattened into the task: {last['task']}"
    assert last["messages"], "no typed turns passed"
    assert {m["role"] for m in last["messages"]} <= {
        "user",
        "assistant",
    }
