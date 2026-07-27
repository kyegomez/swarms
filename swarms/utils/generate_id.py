from typing import Optional
from uuid import uuid4


def generate_id(prefix: Optional[str] = None) -> str:
    """
    Generate a unique identifier for any swarms component.

    The single ID generator for the whole framework. Every ``Agent``, swarm,
    workflow, and conversation routes through this so identifiers share one
    format and one implementation.

    Always call it *inside* ``__init__`` — never as a default argument. Python
    evaluates default arguments once at import, so ``id: str = generate_id()``
    would bake one value into the signature and hand the same ID to every
    instance in the process::

        def __init__(self, id: Optional[str] = None):
            self.id = id or generate_id("agent")

    Args:
        prefix (Optional[str]): Component kind to prefix the ID with, e.g.
            ``"agent"`` or ``"swarm"``. Makes an ID self-describing wherever it
            surfaces — logs, telemetry spans, persisted state. Omit for a bare
            identifier.

    Returns:
        str: ``"<prefix>-<32 hex chars>"``, or just the hex when no prefix is
        given.

    Example:
        >>> generate_id("agent")
        'agent-4cad9d65ef734b22b6be3c31eaa87663'
        >>> generate_id()
        '4cad9d65ef734b22b6be3c31eaa87663'
    """
    unique = uuid4().hex

    return f"{prefix}-{unique}" if prefix else unique
