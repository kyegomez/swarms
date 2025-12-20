import pytest
from swarms.structs.swarm_id import swarm_id


def test_swarm_id_returns_string():
    """Test that swarm_id returns a string"""
    result = swarm_id()
    assert isinstance(result, str)


def test_swarm_id_starts_with_swarm():
    """Test that swarm_id starts with 'swarm-'"""
    result = swarm_id()
    assert result.startswith("swarm-")


def test_swarm_id_has_correct_format():
    """Test that swarm_id has correct format (swarm-{hex})"""
    result = swarm_id()
    parts = result.split("-", 1)
    assert len(parts) == 2
    assert parts[0] == "swarm"
    assert len(parts[1]) == 32  # UUID4 hex is 32 characters


def test_swarm_id_is_unique():
    """Test that swarm_id generates unique IDs"""
    ids = [swarm_id() for _ in range(100)]
    assert len(ids) == len(set(ids))


def test_swarm_id_hex_characters():
    """Test that the hex part contains only valid hex characters"""
    result = swarm_id()
    hex_part = result.split("-", 1)[1]
    assert all(c in "0123456789abcdef" for c in hex_part)


def test_swarm_id_no_hyphens_in_hex():
    """Test that hex part has no hyphens (uuid4().hex strips them)"""
    result = swarm_id()
    hex_part = result.split("-", 1)[1]
    assert "-" not in hex_part
