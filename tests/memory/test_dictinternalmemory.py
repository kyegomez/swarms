# DictInternalMemory

from uuid import uuid4

import pytest

from swarms.memory import DictInternalMemory

# Example of an extensive suite of tests for DictInternalMemory.


# Fixture for repeatedly initializing the class with different numbers of entries.
@pytest.fixture(params=[1, 5, 10, 100])
def memory(request):
    return DictInternalMemory(n_entries=request.param)


# Basic Tests
def test_initialization(memory):
    assert memory.len() == 0


def test_single_add(memory):
    memory.add(10, {"data": "test"})
    assert memory.len() == 1


def test_memory_limit_enforced(memory):
    entries_to_add = memory.n_entries + 10
    for i in range(entries_to_add):
        memory.add(i, {"data": f"test{i}"})
    assert memory.len() == memory.n_entries


# Parameterized Tests
@pytest.mark.parametrize(
    "scores, best_score", [([10, 5, 3], 10), ([1, 2, 3], 3)]
)
def test_get_top_n(scores, best_score, memory):
    for score in scores:
        memory.add(score, {"data": f"test{score}"})
    top_entry = memory.get_top_n(1)
    assert top_entry[0][1]["score"] == best_score


# Exception Testing
@pytest.mark.parametrize("invalid_n", [-1, 0])
def test_invalid_n_entries_raises_exception(invalid_n):
    with pytest.raises(ValueError):
        DictInternalMemory(invalid_n)


# Mocks and Monkeypatching
def test_add_with_mocked_uuid4(monkeypatch, memory):
    # Mock the uuid4 function to return a known value
    class MockUUID:
        hex = "1234abcd"

    monkeypatch.setattr(uuid4, "__str__", lambda: MockUUID.hex)
    memory.add(20, {"data": "mock_uuid"})
    assert MockUUID.hex in memory.data


# Test using Mocks to simulate I/O or external interactions here
# ...

# More tests to hit edge cases, concurrency issues, etc.
# ...

# Tests for concurrency issues, if relevant
# ...
