import pytest


def test_base_swarm_module_can_be_imported():
    """Test that base_swarm module can be imported"""
    from swarms.structs import base_swarm
    assert base_swarm is not None


def test_base_swarm_class_exists():
    """Test that BaseSwarm class exists"""
    from swarms.structs.base_swarm import BaseSwarm
    assert BaseSwarm is not None


def test_base_swarm_is_abstract():
    """Test that BaseSwarm is an abstract base class"""
    from swarms.structs.base_swarm import BaseSwarm
    from abc import ABC
    assert issubclass(BaseSwarm, ABC)
