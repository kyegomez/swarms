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
