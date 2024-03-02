import os
import tempfile

import pytest

from swarms.memory import DictSharedMemory

# Utility functions or fixtures might come first


@pytest.fixture
def memory_file():
    with tempfile.NamedTemporaryFile("w+", delete=False) as tmp_file:
        yield tmp_file.name
    os.unlink(tmp_file.name)


@pytest.fixture
def memory_instance(memory_file):
    return DictSharedMemory(file_loc=memory_file)


# Basic tests


def test_init(memory_file):
    memory = DictSharedMemory(file_loc=memory_file)
    assert os.path.exists(
        memory.file_loc
    ), "Memory file should be created if non-existent"


def test_add_entry(memory_instance):
    success = memory_instance.add(9.5, "agent123", 1, "Test Entry")
    assert success, "add_entry should return True on success"


def test_add_entry_thread_safety(memory_instance):
    # We could create multiple threads to test the thread safety of the add_entry method
    pass


def test_get_top_n(memory_instance):
    memory_instance.add(9.5, "agent123", 1, "Entry A")
    memory_instance.add(8.5, "agent124", 1, "Entry B")
    top_1 = memory_instance.get_top_n(1)
    assert (
        len(top_1) == 1
    ), "get_top_n should return the correct number of top entries"


# Parameterized tests


@pytest.mark.parametrize(
    "scores, agent_ids, expected_top_score",
    [
        ([1.0, 2.0, 3.0], ["agent1", "agent2", "agent3"], 3.0),
        # add more test cases
    ],
)
def test_parametrized_get_top_n(
    memory_instance, scores, agent_ids, expected_top_score
):
    for score, agent_id in zip(scores, agent_ids):
        memory_instance.add(
            score, agent_id, 1, f"Entry by {agent_id}"
        )
    top_1 = memory_instance.get_top_n(1)
    top_score = next(iter(top_1.values()))["score"]
    assert (
        top_score == expected_top_score
    ), "get_top_n should return the entry with top score"


# Exception testing


def test_add_entry_invalid_input(memory_instance):
    with pytest.raises(ValueError):
        memory_instance.add(
            "invalid_score", "agent123", 1, "Test Entry"
        )


# Mocks and monkey-patching


def test_write_fails_due_to_permissions(memory_instance, mocker):
    mocker.patch("builtins.open", side_effect=PermissionError)
    with pytest.raises(PermissionError):
        memory_instance.add(9.5, "agent123", 1, "Test Entry")
