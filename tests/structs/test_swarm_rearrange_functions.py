import pytest
from unittest.mock import Mock
from swarms.structs.swarm_rearrange import swarm_arrange


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
    mock_swarm = Mock()
    mock_swarm.name = "SwarmA"
    mock_swarm.run.return_value = "Result"

    result = swarm_arrange(
        name="TestArrange",
        swarms=[mock_swarm],
        flow="SwarmA",
        task="Test task"
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
    mock_swarm = Mock()
    mock_swarm.name = "SwarmA"
    mock_swarm.run.return_value = "Result"

    result = swarm_arrange(
        name="CustomName",
        description="Custom description",
        swarms=[mock_swarm],
        flow="SwarmA",
        task="Test"
    )
    assert result is not None


def test_swarm_arrange_with_json_output_type():
    """Test swarm_arrange with json output type"""
    mock_swarm = Mock()
    mock_swarm.name = "SwarmA"
    mock_swarm.run.return_value = "Result"

    result = swarm_arrange(
        name="Test",
        swarms=[mock_swarm],
        output_type="json",
        flow="SwarmA",
        task="Test task"
    )
    assert isinstance(result, str)


def test_swarm_arrange_with_default_parameters():
    """Test swarm_arrange with mostly default parameters"""
    result = swarm_arrange()
    assert isinstance(result, str)


def test_swarm_arrange_handles_exceptions():
    """Test that swarm_arrange handles exceptions and returns error string"""
    mock_swarm = Mock()
    mock_swarm.name = "SwarmA"
    mock_swarm.run.side_effect = Exception("Test exception")

    result = swarm_arrange(
        name="Test",
        swarms=[mock_swarm],
        flow="SwarmA",
        task="Test task"
    )
    # Should return error as string
    assert isinstance(result, str)


def test_swarm_arrange_with_multiple_swarms():
    """Test swarm_arrange with multiple swarms"""
    mock_swarm1 = Mock()
    mock_swarm1.name = "SwarmA"
    mock_swarm1.run.return_value = "Result A"

    mock_swarm2 = Mock()
    mock_swarm2.name = "SwarmB"
    mock_swarm2.run.return_value = "Result B"

    result = swarm_arrange(
        name="MultiSwarm",
        swarms=[mock_swarm1, mock_swarm2],
        flow="SwarmA->SwarmB",
        task="Test task"
    )
    assert isinstance(result, str)


def test_swarm_arrange_with_sequential_flow():
    """Test swarm_arrange with sequential flow pattern"""
    mock_swarm1 = Mock()
    mock_swarm1.name = "First"
    mock_swarm1.run.return_value = "First result"

    mock_swarm2 = Mock()
    mock_swarm2.name = "Second"
    mock_swarm2.run.return_value = "Second result"

    result = swarm_arrange(
        name="Sequential",
        swarms=[mock_swarm1, mock_swarm2],
        flow="First->Second",
        task="Start task"
    )
    assert isinstance(result, str)


def test_swarm_arrange_with_kwargs():
    """Test swarm_arrange with additional kwargs"""
    mock_swarm = Mock()
    mock_swarm.name = "SwarmA"
    mock_swarm.run.return_value = "Result"

    result = swarm_arrange(
        name="Test",
        swarms=[mock_swarm],
        flow="SwarmA",
        task="Test",
        custom_param="value"
    )
    assert isinstance(result, str)
