import pytest
from unittest.mock import Mock, patch
from swarms.structs.swarm_arange import SwarmRearrange
from swarms import Agent
from swarm_models import OpenAIChat

@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    return Mock(spec=Agent)

@pytest.fixture
def swarm_rearrange(mock_agent):
    """Create a SwarmRearrange instance with mock agent."""
    return SwarmRearrange(
        id="test_id",
        name="TestSwarm",
        description="Test swarm for testing",
        swarms=[mock_agent],
        flow="Agent1 -> Agent2",
        max_loops=2,
        verbose=True
    )

def test_initialization(swarm_rearrange):
    """Test SwarmRearrange initialization."""
    assert swarm_rearrange.id == "test_id"
    assert swarm_rearrange.name == "TestSwarm"
    assert swarm_rearrange.description == "Test swarm for testing"
    assert len(swarm_rearrange.swarms) == 1
    assert swarm_rearrange.flow == "Agent1 -> Agent2"
    assert swarm_rearrange.max_loops == 2
    assert swarm_rearrange.verbose is True

def test_reliability_checks_empty_swarms():
    """Test reliability checks with empty swarms."""
    with pytest.raises(ValueError, match="No swarms found in the swarm."):
        SwarmRearrange(swarms=[], flow="test")

def test_reliability_checks_empty_flow():
    """Test reliability checks with empty flow."""
    with pytest.raises(ValueError, match="No flow found in the swarm."):
        SwarmRearrange(swarms=[Mock()], flow="")

def test_reliability_checks_invalid_max_loops():
    """Test reliability checks with invalid max_loops."""
    with pytest.raises(ValueError, match="Max loops must be a positive integer."):
        SwarmRearrange(swarms=[Mock()], flow="test", max_loops=0)

def test_add_swarm(swarm_rearrange, mock_agent):
    """Test adding a new swarm."""
    new_agent = Mock(spec=Agent)
    swarm_rearrange.add_swarm(new_agent)
    assert len(swarm_rearrange.swarms) == 2
    assert new_agent in swarm_rearrange.swarms.values()

def test_remove_swarm(swarm_rearrange, mock_agent):
    """Test removing a swarm."""
    swarm_rearrange.remove_swarm(mock_agent.name)
    assert len(swarm_rearrange.swarms) == 0
    assert mock_agent.name not in swarm_rearrange.swarms

def test_add_swarms(swarm_rearrange):
    """Test adding multiple swarms."""
    new_agents = [Mock(spec=Agent) for _ in range(3)]
    swarm_rearrange.add_swarms(new_agents)
    assert len(swarm_rearrange.swarms) == 4
    for agent in new_agents:
        assert agent in swarm_rearrange.swarms.values()

def test_track_history(swarm_rearrange, mock_agent):
    """Test tracking swarm history."""
    result = "Test result"
    swarm_rearrange.track_history(mock_agent.name, result)
    assert result in swarm_rearrange.swarm_history[mock_agent.name]

def test_set_custom_flow(swarm_rearrange):
    """Test setting custom flow."""
    new_flow = "Agent1, Agent2 -> Agent3"
    swarm_rearrange.set_custom_flow(new_flow)
    assert swarm_rearrange.flow == new_flow

def test_context_manager(swarm_rearrange):
    """Test context manager functionality."""
    with swarm_rearrange as db:
        assert db == swarm_rearrange
    # Verify cleanup was performed
    assert not swarm_rearrange.session.is_open()

def test_error_handling(swarm_rearrange):
    """Test error handling in various operations."""
    # Test invalid flow pattern
    with pytest.raises(ValueError):
        swarm_rearrange.set_custom_flow("Invalid -> Flow -> Pattern")
    
    # Test removing non-existent swarm
    with pytest.raises(KeyError):
        swarm_rearrange.remove_swarm("NonExistentSwarm")

def test_thread_safety(swarm_rearrange):
    """Test thread safety of operations."""
    import threading
    import time
    
    def add_swarm_thread():
        for i in range(10):
            new_agent = Mock(spec=Agent)
            new_agent.name = f"Agent{i}"
            swarm_rearrange.add_swarm(new_agent)
            time.sleep(0.1)
    
    def remove_swarm_thread():
        for i in range(10):
            try:
                swarm_rearrange.remove_swarm(f"Agent{i}")
            except KeyError:
                pass
            time.sleep(0.1)
    
    # Create and start threads
    threads = [
        threading.Thread(target=add_swarm_thread),
        threading.Thread(target=remove_swarm_thread)
    ]
    
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # Verify no data corruption occurred
    assert isinstance(swarm_rearrange.swarms, dict)
    assert isinstance(swarm_rearrange.swarm_history, dict) 