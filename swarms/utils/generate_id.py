import secrets
from typing import Optional


def generate_id(prefix: Optional[str] = None) -> str:
    """Generate a unique identifier for any swarms component."""
    return (
        f"{prefix}-{secrets.token_hex(16)}"
        if prefix
        else secrets.token_hex(16)
    )
