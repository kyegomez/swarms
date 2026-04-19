"""Environment loading helpers for Swarms."""

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def resolve_swarms_env_path() -> Optional[Path]:
    """
    Resolve the ``.env`` file used by Swarms.

    Search order:
    1. Current working directory
    2. Parent directories of the current working directory
    3. User home directory
    """
    current_dir = Path.cwd().resolve()

    for directory in (current_dir, *current_dir.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate

    home_candidate = Path.home() / ".env"
    if home_candidate.is_file():
        return home_candidate

    return None


def load_swarms_env(*, override: bool = False) -> bool:
    """Load Swarms environment variables from the resolved ``.env``."""
    dotenv_path = resolve_swarms_env_path()

    if dotenv_path is None:
        return False

    return load_dotenv(dotenv_path=dotenv_path, override=override)
