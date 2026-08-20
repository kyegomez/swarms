import pytest
from swarms.structs.ma_utils import create_agent_map


class MockAgent:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name


class MockAgentWithName:
    def __init__(self, name: str):
        self.name = name


def test_create_agent_map_basic():
    """Test standard non-duplicate, non-rename agent mapping."""
    agent1 = MockAgent("Agent1")
    agent2 = MockAgent("Agent2")
    agent_map = create_agent_map([agent1, agent2])
    assert len(agent_map) == 2
    assert agent_map["Agent1"] is agent1
    assert agent_map["Agent2"] is agent2


def test_create_agent_map_rename_cache_staleness():
    """Regression test for bug 1: renaming an agent reflects the new name on subsequent remap calls."""
    agent = MockAgent("OldName")
    agent_map_1 = create_agent_map([agent])
    assert "OldName" in agent_map_1
    assert agent_map_1["OldName"] is agent

    # Rename agent
    agent.agent_name = "NewName"
    agent_map_2 = create_agent_map([agent])
    assert "NewName" in agent_map_2
    assert "OldName" not in agent_map_2
    assert agent_map_2["NewName"] is agent


def test_create_agent_map_duplicate_names_raises():
    """Regression test for bug 2: duplicate agent names raise ValueError with exact error message."""
    agent1 = MockAgent("DuplicateName")
    agent2 = MockAgent("DuplicateName")

    with pytest.raises(ValueError) as exc_info:
        create_agent_map([agent1, agent2])

    assert "Duplicate agent name 'DuplicateName'" in str(
        exc_info.value
    )
    assert "requires unique agent_name values" in str(exc_info.value)


def test_create_agent_map_plain_callables_and_fallbacks():
    """Test callables without agent_name and objects using name fallback interact correctly with duplicate detection."""

    def func_agent_a():
        pass

    def func_agent_b():
        pass

    # Distinct functions
    agent_map = create_agent_map([func_agent_a, func_agent_b])
    assert "func_agent_a" in agent_map
    assert "func_agent_b" in agent_map
    assert agent_map["func_agent_a"] is func_agent_a

    # Callable and MockAgent with duplicate name
    dup_agent = MockAgent("func_agent_a")
    with pytest.raises(ValueError) as exc_info:
        create_agent_map([func_agent_a, dup_agent])
    assert "Duplicate agent name 'func_agent_a'" in str(
        exc_info.value
    )


def test_create_agent_map_empty_list_raises():
    """Test that passing an empty list raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        create_agent_map([])
    assert "Agents list cannot be empty" in str(exc_info.value)


def test_create_agent_map_missing_name_raises():
    """Test that objects lacking a valid name attribute raise TypeError."""

    class InvalidAgent:
        pass

    with pytest.raises(TypeError) as exc_info:
        create_agent_map([InvalidAgent()])
    assert "lacks required name attribute" in str(exc_info.value)
