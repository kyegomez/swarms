import importlib.util
import os
import runpy
import socket
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = (
    ROOT
    / "examples/single_agent/utils/autosaving_examples"
    / "autosave_directory_structure_example.py"
)


def load_module(monkeypatch, name: str, path: Path) -> ModuleType:
    """Load real workspace code without importing provider packages."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "workspace_name", ["agent_workspace", "custom-workspace"]
)
def test_inspects_each_agents_saved_files(
    monkeypatch, tmp_path: Path, workspace_name: str
) -> None:
    """Inspect actual config files in both supported root layouts."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "WORKSPACE_DIR", str(tmp_path / workspace_name)
    )
    no_network = Mock(side_effect=AssertionError("Network forbidden"))
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)

    swarms = ModuleType("swarms")
    monkeypatch.setitem(sys.modules, "swarms", swarms)
    monkeypatch.setitem(
        sys.modules, "swarms.utils", ModuleType("swarms.utils")
    )
    loguru = ModuleType("loguru")
    loguru.logger = Mock()
    monkeypatch.setitem(sys.modules, "loguru", loguru)
    for name in ("workspace_utils", "workspace_manager"):
        module = load_module(
            monkeypatch,
            f"swarms.utils.{name}",
            ROOT / "swarms/utils" / f"{name}.py",
        )
    manager = module.WorkspaceManager

    class OfflineAgent:
        """Replace only agent setup and model execution."""

        def __init__(self, *, agent_name: str, **kwargs) -> None:
            self.agent_name = agent_name
            self.id = (
                f"agent-00000000-0000-0000-{agent_name[-1:]:0>12}"
            )
            self.workspace = manager.for_agent(self)

        def _get_agent_workspace_dir(self) -> str:
            return self.workspace.dir

        def to_dict(self) -> dict:
            return {"agent_name": self.agent_name}

        def run(self, task: str) -> str:
            self.workspace.save_config()
            Path(self.workspace.dir, "attachments").mkdir()
            return "offline response"

    swarms.Agent = OfflineAgent
    getsize = Mock(wraps=os.path.getsize)
    monkeypatch.setattr(os.path, "getsize", getsize)
    result = runpy.run_path(str(EXAMPLE), run_name="__main__")

    expected = [
        Path(agent.workspace.dir) / "config.json"
        for agent in result["agents"]
    ]
    inspected = [
        Path(call.args[0]) for call in getsize.call_args_list
    ]
    assert len(expected) == 3
    assert all(path.is_file() for path in expected)
    assert inspected == expected
    no_network.assert_not_called()
