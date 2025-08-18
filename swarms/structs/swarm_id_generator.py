import uuid


def generate_swarm_id():
    return f"swarm-{uuid.uuid4().hex}"
