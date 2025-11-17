import pytest

from swarms.structs.agent import Agent
from swarms.structs.swarming_architectures import (
    broadcast,
    circular_swarm,
    exponential_swarm,
    geometric_swarm,
    grid_swarm,
    harmonic_swarm,
    linear_swarm,
    log_swarm,
    mesh_swarm,
    one_to_one,
    one_to_three,
    power_swarm,
    pyramid_swarm,
    sigmoid_swarm,
    sinusoidal_swarm,
    staircase_swarm,
    star_swarm,
)


def create_test_agent(name: str) -> Agent:
    """Create a test agent with specified name"""
    return Agent(
        agent_name=name,
        system_prompt=f"You are {name}. Respond with your name and the task you received.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )


def create_test_agents(num_agents: int) -> list[Agent]:
    """Create specified number of test agents"""
    return [
        create_test_agent(f"Agent{i+1}") for i in range(num_agents)
    ]


def test_circular_swarm():
    """Test circular swarm outputs"""
    agents = create_test_agents(3)
    tasks = [
        "Analyze data",
        "Generate report",
        "Summarize findings",
    ]

    result = circular_swarm(agents, tasks)

    assert isinstance(result, list)
    assert len(result) > 0

    for log in result:
        assert "role" in log
        assert "content" in log


def test_grid_swarm():
    """Test grid swarm with 2x2 grid"""
    agents = create_test_agents(4)
    tasks = ["Task A", "Task B", "Task C", "Task D"]

    result = grid_swarm(agents, tasks)

    assert isinstance(result, list)
    assert len(result) > 0


def test_linear_swarm():
    """Test linear swarm sequential processing"""
    agents = create_test_agents(3)
    tasks = ["Research task", "Write content", "Review output"]

    result = linear_swarm(agents, tasks)

    assert isinstance(result, list)
    assert len(result) > 0

    for log in result:
        assert "role" in log
        assert "content" in log


def test_star_swarm():
    """Test star swarm with central and peripheral agents"""
    agents = create_test_agents(4)
    tasks = ["Coordinate workflow", "Process data"]

    result = star_swarm(agents, tasks)

    assert isinstance(result, list)
    assert len(result) > 0

    for log in result:
        assert "role" in log
        assert "content" in log


def test_mesh_swarm():
    """Test mesh swarm interconnected processing"""
    agents = create_test_agents(3)
    tasks = [
        "Analyze data",
        "Process information",
        "Generate insights",
    ]

    result = mesh_swarm(agents, tasks)

    assert isinstance(result, list)
    assert len(result) > 0

    for log in result:
        assert "role" in log
        assert "content" in log


def test_pyramid_swarm():
    """Test pyramid swarm hierarchical structure"""
    agents = create_test_agents(6)
    tasks = [
        "Top task",
        "Middle task 1",
        "Middle task 2",
        "Bottom task 1",
        "Bottom task 2",
        "Bottom task 3",
    ]

    result = pyramid_swarm(agents, tasks)

    assert isinstance(result, list)
    assert len(result) > 0

    for log in result:
        assert "role" in log
        assert "content" in log


def test_power_swarm():
    """Test power swarm mathematical pattern"""
    agents = create_test_agents(8)
    tasks = [
        "Calculate in Power Swarm",
        "Process in Power Swarm",
        "Analyze in Power Swarm",
    ]

    result = power_swarm(agents, tasks.copy())

    assert isinstance(result, list)
    assert len(result) > 0


def test_log_swarm():
    """Test log swarm mathematical pattern"""
    agents = create_test_agents(8)
    tasks = [
        "Calculate in Log Swarm",
        "Process in Log Swarm",
        "Analyze in Log Swarm",
    ]

    result = log_swarm(agents, tasks.copy())

    assert isinstance(result, list)
    assert len(result) > 0


def test_exponential_swarm():
    """Test exponential swarm mathematical pattern"""
    agents = create_test_agents(8)
    tasks = [
        "Calculate in Exponential Swarm",
        "Process in Exponential Swarm",
        "Analyze in Exponential Swarm",
    ]

    result = exponential_swarm(agents, tasks.copy())

    assert isinstance(result, list)
    assert len(result) > 0


def test_geometric_swarm():
    """Test geometric swarm mathematical pattern"""
    agents = create_test_agents(8)
    tasks = [
        "Calculate in Geometric Swarm",
        "Process in Geometric Swarm",
        "Analyze in Geometric Swarm",
    ]

    result = geometric_swarm(agents, tasks.copy())

    assert isinstance(result, list)
    assert len(result) > 0


def test_harmonic_swarm():
    """Test harmonic swarm mathematical pattern"""
    agents = create_test_agents(8)
    tasks = [
        "Calculate in Harmonic Swarm",
        "Process in Harmonic Swarm",
        "Analyze in Harmonic Swarm",
    ]

    result = harmonic_swarm(agents, tasks.copy())

    assert isinstance(result, list)
    assert len(result) > 0


def test_staircase_swarm():
    """Test staircase swarm pattern"""
    agents = create_test_agents(10)
    tasks = [
        "Process step 1",
        "Process step 2",
        "Process step 3",
        "Process step 4",
        "Process step 5",
    ]

    result = staircase_swarm(agents, tasks)

    assert isinstance(result, list)
    assert len(result) > 0


def test_sigmoid_swarm():
    """Test sigmoid swarm pattern"""
    agents = create_test_agents(10)
    tasks = [
        "Sigmoid task 1",
        "Sigmoid task 2",
        "Sigmoid task 3",
        "Sigmoid task 4",
        "Sigmoid task 5",
    ]

    result = sigmoid_swarm(agents, tasks)

    assert isinstance(result, list)
    assert len(result) > 0


def test_sinusoidal_swarm():
    """Test sinusoidal swarm pattern"""
    agents = create_test_agents(10)
    tasks = [
        "Wave task 1",
        "Wave task 2",
        "Wave task 3",
        "Wave task 4",
        "Wave task 5",
    ]

    result = sinusoidal_swarm(agents, tasks)

    assert isinstance(result, list)
    assert len(result) > 0


def test_one_to_one():
    """Test one-to-one communication pattern"""
    sender = create_test_agent("Sender")
    receiver = create_test_agent("Receiver")
    task = "Process and relay this message"

    result = one_to_one(sender, receiver, task)

    assert isinstance(result, list)
    assert len(result) > 0

    for log in result:
        assert "role" in log
        assert "content" in log


@pytest.mark.asyncio
async def test_one_to_three():
    """Test one-to-three communication pattern"""
    sender = create_test_agent("Sender")
    receivers = create_test_agents(3)
    task = "Process and relay this message"

    result = await one_to_three(sender, receivers, task)

    assert isinstance(result, list)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_broadcast():
    """Test broadcast communication pattern"""
    sender = create_test_agent("Broadcaster")
    receivers = create_test_agents(5)
    task = "Broadcast this message"

    result = await broadcast(sender, receivers, task)

    assert isinstance(result, list)
    assert len(result) > 0
