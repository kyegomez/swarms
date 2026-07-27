from swarms.utils.generate_id import generate_id


def swarm_id() -> str:
    """Deprecated: use ``generate_id("swarm")``."""
    return generate_id("swarm")
