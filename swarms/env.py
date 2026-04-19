"""Environment loading helpers for Swarms."""

import os
import shutil
import subprocess
from io import StringIO
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values, load_dotenv


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


def resolve_dotenvx_binary() -> Optional[str]:
    """Return the available ``dotenvx`` binary, if any."""
    return shutil.which("dotenvx")


def _env_file_looks_encrypted(dotenv_path: Path) -> bool:
    """Detect the markers dotenvx writes into encrypted env files."""
    try:
        contents = dotenv_path.read_text(encoding="utf-8")
    except OSError:
        return False

    return (
        "encrypted:" in contents
        or "DOTENV_PUBLIC_KEY=" in contents
    )


def load_swarms_env_with_dotenvx(
    dotenv_path: Path,
    *,
    override: bool = False,
    dotenvx_binary: Optional[str] = None,
) -> bool:
    """Load env values with ``dotenvx`` using its runtime precedence rules."""
    binary = dotenvx_binary or resolve_dotenvx_binary()
    if binary is None:
        return False

    command = [
        binary,
        "get",
        "--format=eval",
        "-f",
        str(dotenv_path),
    ]
    if override:
        command.append("--overload")

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    values = dotenv_values(stream=StringIO(result.stdout))
    loaded = False

    for key, value in values.items():
        if value is None:
            continue
        os.environ[key] = value
        loaded = True

    return loaded


def load_swarms_env(*, override: bool = False) -> bool:
    """Load Swarms environment variables from the resolved ``.env``."""
    dotenv_path = resolve_swarms_env_path()

    if dotenv_path is None:
        return False

    dotenvx_binary = resolve_dotenvx_binary()

    if dotenvx_binary is not None:
        try:
            return load_swarms_env_with_dotenvx(
                dotenv_path,
                override=override,
                dotenvx_binary=dotenvx_binary,
            )
        except (
            OSError,
            subprocess.CalledProcessError,
        ):
            if _env_file_looks_encrypted(dotenv_path):
                raise

    return load_dotenv(dotenv_path=dotenv_path, override=override)
