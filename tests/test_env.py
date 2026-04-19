import os
import subprocess
from unittest.mock import Mock

import pytest

from swarms.env import (
    load_swarms_env,
    load_swarms_env_with_dotenvx,
    resolve_swarms_env_path,
)


def test_resolve_swarms_env_path_prefers_cwd_tree(tmp_path, monkeypatch):
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    (home_dir / ".env").write_text("OPENAI_API_KEY=home-key\n")

    project_dir = tmp_path / "project"
    nested_dir = project_dir / "nested"
    nested_dir.mkdir(parents=True)
    project_env = project_dir / ".env"
    project_env.write_text("OPENAI_API_KEY=project-key\n")

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.chdir(nested_dir)

    assert resolve_swarms_env_path() == project_env


def test_load_swarms_env_uses_home_fallback(tmp_path, monkeypatch):
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    (home_dir / ".env").write_text("OPENAI_API_KEY=home-key\n")

    project_dir = tmp_path / "project"
    project_dir.mkdir()

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.chdir(project_dir)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert load_swarms_env() is True
    assert os.getenv("OPENAI_API_KEY") == "home-key"


def test_load_swarms_env_prefers_dotenvx_when_available(
    tmp_path, monkeypatch
):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / ".env").write_text("OPENAI_API_KEY=project-key\n")

    monkeypatch.chdir(project_dir)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "swarms.env.resolve_dotenvx_binary",
        lambda: "/usr/local/bin/dotenvx",
    )

    command_result = Mock(stdout='OPENAI_API_KEY="dotenvx-key"\n')
    run_mock = Mock(return_value=command_result)
    monkeypatch.setattr("swarms.env.subprocess.run", run_mock)

    assert load_swarms_env() is True
    assert os.getenv("OPENAI_API_KEY") == "dotenvx-key"
    run_mock.assert_called_once_with(
        [
            "/usr/local/bin/dotenvx",
            "get",
            "--format=eval",
            "-f",
            str(project_dir / ".env"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )


def test_load_swarms_env_with_dotenvx_overrides_when_requested(
    tmp_path, monkeypatch
):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("OPENAI_API_KEY=project-key\n")

    monkeypatch.setenv("OPENAI_API_KEY", "existing")

    command_result = Mock(stdout='OPENAI_API_KEY="dotenvx-key"\n')
    run_mock = Mock(return_value=command_result)
    monkeypatch.setattr("swarms.env.subprocess.run", run_mock)

    assert (
        load_swarms_env_with_dotenvx(
            dotenv_path,
            override=True,
            dotenvx_binary="/usr/local/bin/dotenvx",
        )
        is True
    )
    assert os.getenv("OPENAI_API_KEY") == "dotenvx-key"
    run_mock.assert_called_once_with(
        [
            "/usr/local/bin/dotenvx",
            "get",
            "--format=eval",
            "-f",
            str(dotenv_path),
            "--overload",
        ],
        capture_output=True,
        text=True,
        check=True,
    )


def test_load_swarms_env_falls_back_when_plain_dotenvx_load_fails(
    tmp_path, monkeypatch
):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / ".env").write_text("OPENAI_API_KEY=plain-key\n")

    monkeypatch.chdir(project_dir)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "swarms.env.resolve_dotenvx_binary",
        lambda: "/usr/local/bin/dotenvx",
    )

    error = subprocess.CalledProcessError(
        1,
        ["/usr/local/bin/dotenvx", "get"],
    )
    monkeypatch.setattr(
        "swarms.env.subprocess.run",
        Mock(side_effect=error),
    )

    assert load_swarms_env() is True
    assert os.getenv("OPENAI_API_KEY") == "plain-key"


def test_load_swarms_env_raises_for_encrypted_dotenvx_failure(
    tmp_path, monkeypatch
):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / ".env").write_text(
        "DOTENV_PUBLIC_KEY=public-key\n"
        "OPENAI_API_KEY=encrypted:abc123\n"
    )

    monkeypatch.chdir(project_dir)
    monkeypatch.setattr(
        "swarms.env.resolve_dotenvx_binary",
        lambda: "/usr/local/bin/dotenvx",
    )

    error = subprocess.CalledProcessError(
        1,
        ["/usr/local/bin/dotenvx", "get"],
    )
    monkeypatch.setattr(
        "swarms.env.subprocess.run",
        Mock(side_effect=error),
    )

    with pytest.raises(subprocess.CalledProcessError):
        load_swarms_env()
