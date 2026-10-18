"""Exercise the hiring example with the real agent lookup helper."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/guides/demos/apps/hiring_swarm.py"


class AgentDouble:
    def __init__(self, agent_name: str, **kwargs: object) -> None:
        self.agent_name = agent_name
        self.run = Mock(return_value=f"Reply from {agent_name}")


@pytest.fixture
def hiring_swarm(monkeypatch: pytest.MonkeyPatch) -> tuple:
    conversation = Mock()
    conversation.get_str.return_value = "Previous hiring context"
    formatter = Mock(return_value="Completed hiring history")
    modules = {
        "swarms.structs.agent": SimpleNamespace(Agent=AgentDouble),
        "swarms.structs.conversation": SimpleNamespace(
            Conversation=Mock(return_value=conversation)
        ),
        "swarms.structs.context_utils": SimpleNamespace(
            agent_answer=Mock(), messages_for=Mock()
        ),
        "swarms.structs.multi_agent_exec": SimpleNamespace(
            run_agents_concurrently=Mock()
        ),
        "swarms.utils.history_output_formatter": SimpleNamespace(
            history_output_formatter=formatter,
            HistoryOutputType=str,
        ),
        "swarms.prompts.agent_conversation_aggregator": (
            SimpleNamespace(AGGREGATOR_SYSTEM_PROMPT="Unused")
        ),
        "swarms.structs.ma_utils": SimpleNamespace(
            set_random_models_for_agents=lambda agents: agents
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    blocked = Mock(side_effect=AssertionError("No network"))
    monkeypatch.setattr("socket.socket", blocked)
    monkeypatch.setattr("socket.create_connection", blocked)
    helper = runpy.run_path(str(ROOT / "swarms/structs/ma_blocks.py"))
    monkeypatch.setitem(
        sys.modules,
        "swarms.structs.ma_blocks",
        SimpleNamespace(
            find_agent_by_name=helper["find_agent_by_name"]
        ),
    )
    example = runpy.run_path(str(EXAMPLE))
    return example["HiringSwarm"](), formatter


@pytest.mark.parametrize(
    ("stage", "index"),
    [
        ("initial_talent_acquisition", 0),
        ("candidate_screening", 1),
        ("interview_coordination", 2),
        ("onboarding_preparation", 3),
        ("employee_engagement_strategy", 4),
    ],
)
def test_stage_resolves_its_registered_agent(
    hiring_swarm: tuple, stage: str, index: int
) -> None:
    swarm, _ = hiring_swarm
    target = swarm.agents[index]
    swarm.agents.reverse()
    getattr(swarm, stage)()
    target.run.assert_called_once()
    for agent in swarm.agents:
        if agent is not target:
            agent.run.assert_not_called()


def test_default_workflow_reaches_every_stage(
    hiring_swarm: tuple,
) -> None:
    swarm, formatter = hiring_swarm
    assert swarm.run("Hire a software engineer") == (
        "Completed hiring history"
    )
    for agent in swarm.agents:
        agent.run.assert_called_once()
    formatter.assert_called_once_with(swarm.conversation, type="list")
