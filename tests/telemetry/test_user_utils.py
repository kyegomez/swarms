import uuid

from swarms.telemetry.main import (
    generate_user_id,
    get_machine_id,
)


# Helper functions tests
def test_generate_user_id():
    # Generate user IDs and ensure they are UUID strings
    user_id = generate_user_id()
    assert isinstance(user_id, str)
    assert uuid.UUID(user_id, version=4)


def test_get_machine_id():
    # Get machine ID and ensure it's a valid SHA-256 hash
    machine_id = get_machine_id()
    assert isinstance(machine_id, str)
    assert len(machine_id) == 64
    assert all(char in "0123456789abcdef" for char in machine_id)


def test_generate_user_id_edge_case():
    # Test generate_user_id with multiple calls
    user_ids = set()
    for _ in range(100):
        user_id = generate_user_id()
        user_ids.add(user_id)
    assert len(user_ids) == 100  # Ensure generated IDs are unique


def test_get_machine_id_edge_case():
    # Test get_machine_id with multiple calls
    machine_ids = set()
    for _ in range(100):
        machine_id = get_machine_id()
        machine_ids.add(machine_id)
    assert len(machine_ids) == 100  # Ensure generated IDs are unique



def test_all():
    test_generate_user_id()
    test_get_machine_id()
    test_generate_user_id_edge_case()
    test_get_machine_id_edge_case()



test_all()
