from uuid import uuid4


def swarm_id():
    return f"swarm-{uuid4().hex}"
