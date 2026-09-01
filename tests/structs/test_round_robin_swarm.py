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

    assert second.calls, "second agent never ran"
    last = second.calls[-1]

    assert (
        "A-out" not in last["task"]
    ), f"transcript flattened into the task: {last['task']}"
    assert last["messages"], "no typed turns passed"
    assert {m["role"] for m in last["messages"]} <= {
        "user",
        "assistant",
    }

    contents = [m["content"] for m in last["messages"]]
    assert any(
        "A-out" in text for text in contents
    ), f"prior speaker's output missing from the turns: {contents}"
    assert any(
        "original task" in text for text in contents
    ), f"original task missing from the turns: {contents}"


class RecordingAgent:
    """Stands in for an Agent: named, and it appends itself to a log."""

    def __init__(self, name, log):
        self.agent_name = name
        self._log = log

    def run(self, task, *args, **kwargs):
        self._log.append(self.agent_name)
        return f"{self.agent_name} responded"


def build_swarm(n=3, max_loops=1):
    log = []
    agents = [RecordingAgent(f"Agent{i}", log) for i in range(n)]
    swarm = RoundRobinSwarm(
        agents=agents, max_loops=max_loops, output_type="final"
    )
    return swarm, log


def test_a_single_run_visits_every_agent_once_per_loop():
    swarm, log = build_swarm(n=3, max_loops=2)
    swarm.run("t")
    assert log == [
        "Agent0",
        "Agent1",
        "Agent2",
        "Agent0",
        "Agent1",
        "Agent2",
    ]


def test_every_agent_opens_exactly_once_across_a_batch():
    """The opener frames the task, so the seat has to move between runs.

    K * N turns is a whole number of rotations, so the offset returns to where
    it started: without an explicit step, agents[0] opened every task in the
    batch and agents[-1] opened none.
    """
    swarm, log = build_swarm(n=3, max_loops=1)
    swarm.run_batch(["a", "b", "c"])

    openers = [log[i * 3] for i in range(3)]
    assert sorted(openers) == ["Agent0", "Agent1", "Agent2"]
    assert openers == ["Agent0", "Agent1", "Agent2"]


def test_the_rotation_wraps_and_stays_a_full_cycle():
    swarm, log = build_swarm(n=3, max_loops=1)
    for _ in range(4):
        log.clear()
        swarm.run("t")
        assert sorted(log) == ["Agent0", "Agent1", "Agent2"], (
            "every agent must still get exactly one turn per loop, "
            f"got {log}"
        )
    # Four runs over three agents: back to the start, one further along.
    assert swarm.index == 1
    assert log[0] == "Agent0"


@pytest.mark.parametrize("n", [1, 2, 5])
def test_turns_stay_evenly_distributed_for_any_roster_size(n):
    swarm, log = build_swarm(n=n, max_loops=3)
    swarm.run("t")
    assert len(log) == n * 3
    assert all(log.count(a.agent_name) == 3 for a in swarm.agents)
