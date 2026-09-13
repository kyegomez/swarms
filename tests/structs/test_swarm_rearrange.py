from unittest.mock import MagicMock
import pytest

from swarms.structs.swarm_rearrange import SwarmRearrange


def _make_swarm(name: str):
    swarm = MagicMock()
    swarm.name = name

    def _run(task, *args, **kwargs):
        return f"{name}:{task}"

    swarm.run = _run
    return swarm


def test_swarm_rearrange_init():
    s1 = _make_swarm("Swarm-1")
    s2 = _make_swarm("Swarm-2")
    sr = SwarmRearrange(name="TestSR", swarms=[s1, s2], flow="Swarm-1 -> Swarm-2")
    assert sr.name == "TestSR"
    assert len(sr.swarms) == 2
    assert sr.flow == "Swarm-1 -> Swarm-2"


def test_swarm_rearrange_run():
    s1 = _make_swarm("Swarm-1")
    s2 = _make_swarm("Swarm-2")
    sr = SwarmRearrange(name="TestSR", swarms=[s1, s2], flow="Swarm-1 -> Swarm-2")
    result = sr.run("hello")
    assert "Swarm-2" in result and "Swarm-1" in result
