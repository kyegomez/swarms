import pytest

from swarms.structs.agent import Agent
from swarms.structs.swarming_architectures import (
    broadcast,
    circular_swarm,
    grid_swarm,
    mesh_swarm,
    one_to_one,
    pyramid_swarm,
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
async def test_broadcast():
    """Test broadcast communication pattern"""
    sender = create_test_agent("Broadcaster")
    receivers = create_test_agents(5)
    task = "Broadcast this message"

    result = await broadcast(sender, receivers, task)

    assert isinstance(result, list)
    assert len(result) > 0
