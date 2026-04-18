"""Environment loading helpers for Swarms."""

from dotenv import find_dotenv, load_dotenv


def load_swarms_env(*, override: bool = False) -> bool:
    """Load a ``.env`` file by searching upward from the current working directory."""
    dotenv_path = find_dotenv(".env", usecwd=True)

    if not dotenv_path:
        return False

    return load_dotenv(dotenv_path=dotenv_path, override=override)
