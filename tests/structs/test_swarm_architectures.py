import os
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
        model_name="gpt-5.4",
        max_loops=1,
    )


def create_test_agents(num_agents: int) -> list[Agent]:
    """Create specified number of test agents"""
    return [
        create_test_agent(f"Agent{i+1}") for i in range(num_agents)
    ]



# Live-LLM tests: they call agent.run() against a real model and can only
# pass with an API key. Without one, Agent.run now honestly raises
# AgentLLMError after retry exhaustion, so these tests are skipped instead
# of failing (same convention as tests/telemetry/test_telemetry.py).
_LLM_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
)
_HAS_LLM_KEY = any(os.getenv(k) for k in _LLM_KEYS)
requires_llm = pytest.mark.skipif(
    not _HAS_LLM_KEY,
    reason="no LLM API key set (OPENAI_API_KEY etc.) - live-LLM test skipped",
)

@requires_llm
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


@requires_llm
def test_grid_swarm():
    """Test grid swarm with 2x2 grid"""
    agents = create_test_agents(4)
    tasks = ["Task A", "Task B", "Task C", "Task D"]

    result = grid_swarm(agents, tasks)

    assert isinstance(result, list)
    assert len(result) > 0


@requires_llm
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


@requires_llm
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


@requires_llm
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


@requires_llm
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
