import pytest
from swarms.structs.swarm_rearrange import swarm_arrange
from swarms.structs.agent import Agent
from swarms.structs.swarm_router import SwarmRouter


def create_test_agent(name: str, description: str = "Test agent") -> Agent:
    """Create a real Agent instance for testing"""
    return Agent(
        agent_name=name,
        agent_description=description,
        system_prompt=f"You are {name}, a helpful test assistant. Keep responses brief.",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
    )


def create_test_swarm(name: str) -> SwarmRouter:
    """Create a real SwarmRouter instance for testing"""
    agent = create_test_agent(f"{name}_agent")
    return SwarmRouter(
        name=name,
        description=f"Test swarm {name}",
        agents=[agent],
        swarm_type="SequentialWorkflow",
        max_loops=1,
    )


def test_swarm_arrange_with_none_swarms():
    """Test swarm_arrange with None swarms parameter"""
    result = swarm_arrange(
        name="Test",
        swarms=None,
        flow="A->B",
        task="Test task"
    )
    # Should handle None swarms gracefully
    assert result is not None


def test_swarm_arrange_returns_string():
    """Test that swarm_arrange returns a string"""
    swarm = create_test_swarm("SwarmA")

    result = swarm_arrange(
        name="TestArrange",
        swarms=[swarm],
        flow="SwarmA",
        task="What is 2+2?"
    )
    assert isinstance(result, str)


def test_swarm_arrange_with_empty_swarms_list():
    """Test swarm_arrange with empty swarms list"""
    result = swarm_arrange(
        name="Test",
        swarms=[],
        flow="A->B",
        task="Test task"
    )
    # Should handle empty swarms
    assert isinstance(result, str)


def test_swarm_arrange_with_custom_name():
    """Test swarm_arrange with custom name"""
    swarm = create_test_swarm("SwarmA")

    result = swarm_arrange(
        name="CustomName",
        description="Custom description",
        swarms=[swarm],
        flow="SwarmA",
        task="Say hello"
    )
    assert result is not None


def test_swarm_arrange_with_json_output_type():
    """Test swarm_arrange with json output type"""
    swarm = create_test_swarm("SwarmA")

    result = swarm_arrange(
        name="Test",
        swarms=[swarm],
        output_type="json",
        flow="SwarmA",
        task="What is 1+1?"
    )
    assert isinstance(result, str)


def test_swarm_arrange_with_default_parameters():
    """Test swarm_arrange with mostly default parameters"""
    result = swarm_arrange()
    assert isinstance(result, str)


def test_swarm_arrange_with_multiple_swarms():
    """Test swarm_arrange with multiple swarms"""
    swarm1 = create_test_swarm("SwarmA")
    swarm2 = create_test_swarm("SwarmB")

    result = swarm_arrange(
        name="MultiSwarm",
        swarms=[swarm1, swarm2],
        flow="SwarmA->SwarmB",
        task="Complete this simple task"
    )
    assert isinstance(result, str)


def test_swarm_arrange_with_sequential_flow():
    """Test swarm_arrange with sequential flow pattern"""
    swarm1 = create_test_swarm("First")
    swarm2 = create_test_swarm("Second")

    result = swarm_arrange(
        name="Sequential",
        swarms=[swarm1, swarm2],
        flow="First->Second",
        task="Process this step by step"
    )
    assert isinstance(result, str)


def test_swarm_arrange_with_kwargs():
    """Test swarm_arrange with additional kwargs"""
    swarm = create_test_swarm("SwarmA")

    result = swarm_arrange(
        name="Test",
        swarms=[swarm],
        flow="SwarmA",
        task="Simple test"
    )
    assert isinstance(result, str)
