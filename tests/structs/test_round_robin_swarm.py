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


def _visit_order(swarm, task="t"):
    """Agent names in the order they were given a turn, without LLM calls."""
    from unittest.mock import patch

    seen = []
    with patch.object(
        swarm,
        "_execute_agent",
        side_effect=lambda agent, *a, **k: seen.append(
            agent.agent_name
        ),
    ):
        swarm.run(task)
    return seen


def test_rotation_does_not_persist_by_default():
    """Default stays turn-for-turn reproducible.

    `persist_rotation` is opt-in because callers rely on run() replaying the
    same order; this pins that the default did not change.
    """
    agents = [Agent(agent_name=f"A{i}") for i in range(3)]
    swarm = RoundRobinSwarm(agents=agents, max_loops=2)

    first = _visit_order(swarm)
    assert _visit_order(swarm) == first
    assert first[0] == "A0"


def test_persist_rotation_gives_every_agent_the_opening_turn():
    """`self.index` was written every turn and read by nothing, so every run()
    restarted at agents[0] — run_batch handed the opening turn, which frames
    the transcript every later agent builds on, to the same agent for every
    task in the batch.
    """
    agents = [Agent(agent_name=f"A{i}") for i in range(3)]
    swarm = RoundRobinSwarm(
        agents=agents, max_loops=2, persist_rotation=True
    )

    runs = [_visit_order(swarm) for _ in range(3)]

    # Each task opens with a different agent.
    assert [r[0] for r in runs] == ["A0", "A1", "A2"]

    # And within a run every agent still gets exactly max_loops turns.
    for r in runs:
        assert len(r) == 6
        assert {name: r.count(name) for name in set(r)} == {
            "A0": 2,
            "A1": 2,
            "A2": 2,
        }
