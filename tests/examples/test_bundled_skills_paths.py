"""Exercise bundled skill discovery independently of the working directory."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
SKILLS = ROOT / "examples/single_agent/capabilities/skills"


@pytest.fixture
def skills_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules, "loguru", SimpleNamespace(logger=Mock())
    )
    loader = runpy.run_path(
        str(ROOT / "swarms/structs/dynamic_skills_loader.py")
    )
    monkeypatch.setitem(
        sys.modules,
        "swarms.structs.dynamic_skills_loader",
        SimpleNamespace(**loader),
    )
    manager = runpy.run_path(
        str(ROOT / "swarms/agents/skills_manager.py")
    )["SkillsManager"]

    class OfflineAgent:
        def __init__(self, skills_dir: str, **kwargs) -> None:
            self.skills = manager(skills_dir=skills_dir)

        def run(self, task: str) -> str:
            return self.skills.prompt_for_task(task)

    monkeypatch.setitem(
        sys.modules, "swarms", SimpleNamespace(Agent=OfflineAgent)
    )
    blocked = Mock(side_effect=AssertionError("No network"))
    monkeypatch.setattr("socket.socket", blocked)
    monkeypatch.setattr("socket.create_connection", blocked)


@pytest.mark.parametrize("working_directory", ["root", "unrelated"])
@pytest.mark.parametrize(
    ("path", "expected_skill"),
    [
        ("examples/utils/skills_load.py", "data-visualization"),
        (
            "examples/guides/changelog_890/01_dynamic_skills_loader.py",
            "financial-analysis",
        ),
        (
            (
                "examples/single_agent/capabilities/skills/"
                "agent_with_dynamic_skills.py"
            ),
            "financial-analysis",
        ),
    ],
)
def test_examples_load_bundled_skill_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    skills_modules: None,
    working_directory: str,
    path: str,
    expected_skill: str,
) -> None:
    monkeypatch.chdir(
        ROOT if working_directory == "root" else tmp_path
    )
    namespace = runpy.run_path(str(ROOT / path))
    output = capsys.readouterr().out
    if "agent" in namespace:
        selected = namespace["agent"].skills.metadata
        assert f"## {expected_skill}\n" in output
    else:
        selected = json.loads(output)

    assert [skill["name"] for skill in selected] == [expected_skill]
    skill = selected[0]
    expected_file = SKILLS / expected_skill / "SKILL.md"
    assert Path(skill["path"]).resolve() == expected_file
    assert (
        skill["content"]
        == expected_file.read_text().split("---", 2)[2].strip()
    )
    assert skill["similarity_score"] >= (
        0.4 if expected_skill == "data-visualization" else 0.3
    )
