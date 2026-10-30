import pytest
from swarms.structs.auto_swarm_builder import AutoSwarmBuilder


def test_auto_swarm_builder_accepts_init_args():
    """AutoSwarmBuilder must accept max_agents and max_json_repair_attempts."""
    builder = AutoSwarmBuilder(max_agents=3, max_json_repair_attempts=2)
    assert builder.max_agents == 3
    assert builder.max_json_repair_attempts == 2


def test_auto_swarm_builder_default_init_args():
    """AutoSwarmBuilder defaults max_agents to None and max_json_repair_attempts to 1."""
    builder = AutoSwarmBuilder()
    assert builder.max_agents is None
    assert builder.max_json_repair_attempts == 1


def test_reliability_check_rejects_non_positive_max_agents():
    """AutoSwarmBuilder rejects max_agents <= 0."""
    with pytest.raises(ValueError, match="max_agents must be greater than 0"):
        AutoSwarmBuilder(max_agents=0)

    with pytest.raises(ValueError, match="max_agents must be greater than 0"):
        AutoSwarmBuilder(max_agents=-5)


def test_reliability_check_rejects_negative_max_json_repair_attempts():
    """AutoSwarmBuilder rejects max_json_repair_attempts < 0."""
    with pytest.raises(ValueError, match="max_json_repair_attempts must be greater than or equal to 0"):
        AutoSwarmBuilder(max_json_repair_attempts=-1)
