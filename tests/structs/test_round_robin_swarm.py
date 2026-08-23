"""Scheduling tests for RoundRobinSwarm.

Offline: the swarm's only outward call is ``agent.run``, so a stub agent that
records its own name is enough to read the visit order straight off the
schedule. The previous ``test_run`` here built real ``Agent`` objects and hit
a provider, so it failed with a credentials error rather than telling anyone
anything about the rotation.
"""

import pytest

from swarms.structs.round_robin import RoundRobinSwarm


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


def test_init():
    swarm, _ = build_swarm(max_loops=2)
    assert isinstance(swarm, RoundRobinSwarm)
    assert swarm.max_loops == 2
    assert len(swarm.agents) == 3
    assert swarm.index == 0


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
