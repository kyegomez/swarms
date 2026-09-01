from unittest.mock import patch

import pytest

from swarms.structs.agent import Agent
from swarms.structs.round_robin import RoundRobinSwarm


@pytest.fixture
def round_robin_swarm():
    agents = [Agent(agent_name=f"Agent{i}") for i in range(3)]
    return RoundRobinSwarm(agents=agents, verbose=True, max_loops=2)


def test_init(round_robin_swarm):
    assert isinstance(round_robin_swarm, RoundRobinSwarm)
    assert round_robin_swarm.verbose is True
    assert round_robin_swarm.max_loops == 2
    assert round_robin_swarm.persist_rotation is False
    assert len(round_robin_swarm.agents) == 3


def _visit_order(swarm, task="t"):
    """Agent names in turn order without LLM calls."""
    seen = []

    def record(agent, *_args, **_kwargs):
        seen.append(agent.agent_name)
        return f"{agent.agent_name}-ok"

    with patch.object(swarm, "_execute_agent", side_effect=record):
        swarm.run(task)
    return seen


def test_rotation_does_not_persist_by_default():
    """Default stays turn-for-turn reproducible across consecutive run() calls."""
    agents = [Agent(agent_name=f"A{i}") for i in range(3)]
    swarm = RoundRobinSwarm(agents=agents, max_loops=2)

    first = _visit_order(swarm)
    assert _visit_order(swarm) == first
    assert first[0] == "A0"
    assert swarm.index == 0


def test_persist_rotation_run_batch_distributes_opening_turn():
    """N tasks on N agents: each agent opens exactly once when rotation persists."""
    agents = [Agent(agent_name=f"A{i}") for i in range(3)]
    swarm = RoundRobinSwarm(
        agents=agents, max_loops=1, persist_rotation=True
    )
    n = len(agents)

    all_calls = []

    def record(agent, *_args, **_kwargs):
        all_calls.append(agent.agent_name)
        return "ok"

    with patch.object(swarm, "_execute_agent", side_effect=record):
        swarm.run_batch(["task-1", "task-2", "task-3"])

    openers = [all_calls[i * n] for i in range(n)]
    assert openers == ["A0", "A1", "A2"]


def test_persist_rotation_false_identical_visit_order():
    """With the flag off, two consecutive run() calls produce the same order."""
    agents = [Agent(agent_name=f"A{i}") for i in range(3)]
    swarm = RoundRobinSwarm(
        agents=agents, max_loops=2, persist_rotation=False
    )

    first = _visit_order(swarm)
    second = _visit_order(swarm)

    assert first == second
    assert first[0] == "A0"
    assert swarm.index == 0


def test_within_run_every_agent_gets_max_loops_turns():
    agents = [Agent(agent_name=f"A{i}") for i in range(3)]
    swarm = RoundRobinSwarm(
        agents=agents, max_loops=2, persist_rotation=True
    )

    order = _visit_order(swarm)

    assert len(order) == 6
    assert {
        name: order.count(name) for name in ("A0", "A1", "A2")
    } == {
        "A0": 2,
        "A1": 2,
        "A2": 2,
    }
