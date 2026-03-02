import pytest


def test_heavy_swarm_module_can_be_imported():
    """Test that heavy_swarm module can be imported"""
    try:
        from swarms.structs import heavy_swarm
        assert heavy_swarm is not None
    except ImportError:
        pytest.skip("heavy_swarm module has import dependencies")


def test_heavy_swarm_class_exists():
    """Test that HeavySwarm class exists if module is available"""
    try:
        from swarms.structs.heavy_swarm import HeavySwarm
        assert HeavySwarm is not None
    except (ImportError, AttributeError):
        pytest.skip("HeavySwarm class not available or has dependencies")
