import pytest
from unittest.mock import Mock
from swarms.structs.swarming_architectures import (
    circular_swarm,
    grid_swarm,
    linear_swarm,
    star_swarm,
    mesh_swarm,
    pyramid_swarm,
    fibonacci_swarm,
    prime_swarm,
    power_swarm,
    log_swarm,
    exponential_swarm,
    geometric_swarm,
    harmonic_swarm,
    staircase_swarm,
    sigmoid_swarm,
    sinusoidal_swarm,
    one_to_one,
)


def test_circular_swarm_with_single_agent():
    """Test circular_swarm with a single agent"""
    agent = Mock()
    agent.run.return_value = "Response"

    result = circular_swarm(agents=[agent], tasks=["Task 1"])
    assert result is not None


def test_grid_swarm_with_agents():
    """Test grid_swarm with multiple agents"""
    agents = [Mock() for _ in range(4)]
    for agent in agents:
        agent.run.return_value = "Response"

    result = grid_swarm(agents=agents, tasks=["Task 1", "Task 2"])
    assert result is not None


def test_linear_swarm_with_tasks():
    """Test linear_swarm with sequential tasks"""
    agents = [Mock(), Mock()]
    for agent in agents:
        agent.run.return_value = "Response"

    result = linear_swarm(agents=agents, tasks=["Task 1"])
    assert result is not None


def test_star_swarm_basic():
    """Test star_swarm with central coordinator"""
    agents = [Mock() for _ in range(3)]
    for agent in agents:
        agent.run.return_value = "Response"

    result = star_swarm(agents=agents, tasks=["Task"])
    assert result is not None


def test_mesh_swarm_basic():
    """Test mesh_swarm with interconnected agents"""
    agents = [Mock(), Mock()]
    for agent in agents:
        agent.run.return_value = "Response"

    result = mesh_swarm(agents=agents, tasks=["Task"])
    assert result is not None


def test_pyramid_swarm_basic():
    """Test pyramid_swarm hierarchical structure"""
    agents = [Mock() for _ in range(3)]
    for agent in agents:
        agent.run.return_value = "Response"

    result = pyramid_swarm(agents=agents, tasks=["Task"])
    assert result is not None


def test_fibonacci_swarm_basic():
    """Test fibonacci_swarm with fibonacci sequence"""
    agents = [Mock() for _ in range(5)]
    for agent in agents:
        agent.run.return_value = "Response"

    result = fibonacci_swarm(agents=agents, tasks=["Task"])
    assert result is not None


def test_prime_swarm_basic():
    """Test prime_swarm with prime number sequence"""
    agents = [Mock() for _ in range(5)]
    for agent in agents:
        agent.run.return_value = "Response"

    result = prime_swarm(agents=agents, tasks=["Task"])
    assert result is not None


def test_power_swarm_basic():
    """Test power_swarm with power sequence"""
    agents = [Mock() for _ in range(4)]
    for agent in agents:
        agent.run.return_value = "Response"

    result = power_swarm(agents=agents, tasks=["Task"])
    assert result is not None


def test_log_swarm_basic():
    """Test log_swarm with logarithmic sequence"""
    agents = [Mock() for _ in range(3)]
    for agent in agents:
        agent.run.return_value = "Response"

    result = log_swarm(agents=agents, tasks=["Task"])
    assert result is not None


def test_exponential_swarm_basic():
    """Test exponential_swarm with exponential growth"""
    agents = [Mock() for _ in range(4)]
    for agent in agents:
        agent.run.return_value = "Response"

    result = exponential_swarm(agents=agents, tasks=["Task"])
    assert result is not None


def test_geometric_swarm_basic():
    """Test geometric_swarm with geometric sequence"""
    agents = [Mock() for _ in range(4)]
    for agent in agents:
        agent.run.return_value = "Response"

    result = geometric_swarm(agents=agents, tasks=["Task"])
    assert result is not None


def test_harmonic_swarm_basic():
    """Test harmonic_swarm with harmonic sequence"""
    agents = [Mock() for _ in range(3)]
    for agent in agents:
        agent.run.return_value = "Response"

    result = harmonic_swarm(agents=agents, tasks=["Task"])
    assert result is not None


def test_sigmoid_swarm_basic():
    """Test sigmoid_swarm with sigmoid curve"""
    agents = [Mock() for _ in range(3)]
    for agent in agents:
        agent.run.return_value = "Response"

    result = sigmoid_swarm(agents=agents, tasks=["Task"])
    assert result is not None


def test_sinusoidal_swarm_basic():
    """Test sinusoidal_swarm with sinusoidal pattern"""
    agents = [Mock() for _ in range(3)]
    for agent in agents:
        agent.run.return_value = "Response"

    result = sinusoidal_swarm(agents=agents, tasks=["Task"])
    assert result is not None


def test_circular_swarm_returns_result():
    """Test that circular_swarm returns a result"""
    agent = Mock()
    agent.run.return_value = "Test response"

    result = circular_swarm(agents=[agent], tasks=["Task"])
    assert result is not None


def test_linear_swarm_returns_result():
    """Test that linear_swarm returns a result"""
    agent = Mock()
    agent.run.return_value = "Test response"

    result = linear_swarm(agents=[agent], tasks=["Task"])
    assert result is not None
