"""Load the bundled advisor through the real parser from another cwd."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples/utils/agent_loader"


@pytest.mark.parametrize(
    "example", ["agent_loader_demo.py", "claude_code_compatible.py"]
)
@pytest.mark.parametrize(
    "cwd", ["repository", "unrelated", "example"]
)
def test_bundled_advisor_loads_outside_example_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    example: str,
    cwd: str,
) -> None:
    agents = []

    def make_agent(**kwargs: object) -> SimpleNamespace:
        agent = SimpleNamespace(**kwargs, run=Mock(return_value="ok"))
        agents.append(agent)
        return agent

    monkeypatch.setitem(
        sys.modules, "loguru", SimpleNamespace(logger=Mock())
    )
    monkeypatch.setitem(
        sys.modules,
        "swarms.structs.agent",
        SimpleNamespace(Agent=make_agent),
    )
    blocked = Mock(side_effect=AssertionError("No network"))
    monkeypatch.setattr("socket.socket", blocked)
    monkeypatch.setattr("socket.create_connection", blocked)
    loader = runpy.run_path(
        str(ROOT / "swarms/utils/agent_loader_markdown.py")
    )
    exports = SimpleNamespace(
        load_agent_from_markdown=loader["load_agent_from_markdown"],
        load_agents_from_markdown=loader["load_agents_from_markdown"],
    )
    monkeypatch.setitem(sys.modules, "swarms", exports)
    monkeypatch.setitem(sys.modules, "swarms.utils", exports)
    monkeypatch.chdir(
        {
            "repository": ROOT,
            "unrelated": tmp_path,
            "example": EXAMPLES,
        }[cwd]
    )
    runpy.run_path(str(EXAMPLES / example))
    assert len(agents) == 1
    assert agents[0].agent_name == "FinanceAdvisor"
    assert (
        "You are an expert financial advisor"
        in agents[0].system_prompt
    )
    agents[0].run.assert_called_once()
