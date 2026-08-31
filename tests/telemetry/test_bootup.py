import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

PROBE = """
import json, os
import swarms
from swarms.utils.loguru_logger import get_log_dir
print(json.dumps({
    "workspace": os.environ.get("WORKSPACE_DIR"),
    "log_dir": get_log_dir(),
}))
"""


def run_probe(cwd, env_overrides):
    """Import swarms in a fresh interpreter and report what it resolved."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.pop("WORKSPACE_DIR", None)
    for key, value in env_overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value

    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert (
        result.returncode == 0
    ), f"import swarms failed:\n{result.stderr}"

    # The package logs to stdout, so take the last line, which is the probe's.
    payload = [
        line for line in result.stdout.splitlines() if line.strip()
    ][-1]
    return json.loads(payload)


def test_exported_workspace_dir_survives_import(tmp_path):
    """The reported bug: an exported value was discarded by bootup."""
    target = tmp_path / "mydir"

    out = run_probe(tmp_path, {"WORKSPACE_DIR": str(target)})

    assert out["workspace"] == str(target)
    assert out["log_dir"] == os.path.join(str(target), "logs")
    assert target.is_dir()


def test_unset_workspace_dir_defaults_under_cwd(tmp_path):
    """Unset stays defaulted, and the default is absolute rather than a bare
    relative name that would follow the process around."""
    out = run_probe(tmp_path, {"WORKSPACE_DIR": None})

    workspace = Path(out["workspace"])
    assert workspace.is_absolute()
    assert workspace.name == "agent_workspace"
    assert workspace.parent.resolve() == tmp_path.resolve()


def test_relative_workspace_dir_is_left_alone(tmp_path):
    """A relative value is the caller's choice; it is not rewritten."""
    out = run_probe(tmp_path, {"WORKSPACE_DIR": "my_ws"})

    assert out["workspace"] == "my_ws"
    assert (tmp_path / "my_ws").is_dir()


@pytest.mark.parametrize("kind", ["file", "unwritable"])
def test_unusable_workspace_dir_does_not_break_the_import(
    tmp_path, kind
):
    """An unusable WORKSPACE_DIR degrades to the default.

    ``mkdir`` runs against a caller-supplied path inside a ``try`` block that
    re-raises, so without a guard a path that cannot be created takes down
    ``import swarms`` entirely.
    """
    if kind == "file":
        # exist_ok does not suppress FileExistsError when the path is a file.
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory")
        value = str(blocker)
    else:
        parent = tmp_path / "locked"
        parent.mkdir()
        parent.chmod(0o500)
        value = str(parent / "child")

    try:
        out = run_probe(tmp_path, {"WORKSPACE_DIR": value})
    finally:
        if kind == "unwritable":
            (tmp_path / "locked").chmod(0o700)

    # Fell back rather than raising, and the fallback is usable.
    assert out["workspace"] != value
    assert Path(out["workspace"]).is_dir()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
