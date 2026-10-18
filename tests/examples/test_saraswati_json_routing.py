"""Run the example's router with JSON replies and real Pydantic."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples/guides/demos/hackathon_feb16/sarasowti.py"
)


@pytest.fixture
def example(monkeypatch: pytest.MonkeyPatch) -> tuple:
    agents = {}
    model = Mock()

    def create_agent(**kwargs: object) -> SimpleNamespace:
        name = kwargs["agent_name"]
        agent = SimpleNamespace(
            agent_name=name, run=Mock(return_value="Offline advice")
        )
        agents[name] = agent
        return agent

    def find_agent(registered: list, name: str) -> object:
        return next(a for a in registered if a.agent_name == name)

    for name, module in {
        "dotenv": SimpleNamespace(load_dotenv=Mock()),
        "swarms": SimpleNamespace(Agent=create_agent),
        "swarms.utils.litellm_wrapper": SimpleNamespace(
            LiteLLM=Mock(return_value=model)
        ),
        "swarms.structs.conversation": SimpleNamespace(
            Conversation=Mock
        ),
        "swarms.structs.ma_blocks": SimpleNamespace(
            find_agent_by_name=find_agent
        ),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    blocked = Mock(side_effect=AssertionError("No network"))
    monkeypatch.setattr("socket.socket", blocked)
    monkeypatch.setattr("socket.create_connection", blocked)

    def run_example(response: str) -> dict:
        model.run.return_value = response
        return runpy.run_path(str(EXAMPLE))

    return run_example, agents


@pytest.mark.parametrize("target", ["Counselor-Agent", "Buddy-Agent"])
def test_json_response_dispatches_to_selected_agent(
    example: tuple, target: str
) -> None:
    run_example, agents = example
    task = "Discuss the student's college options — 学业规划"
    namespace = run_example(
        json.dumps(
            {
                "response_to_user": "I will ask a specialist.",
                "agent_name": target,
                "task": task,
            }
        )
    )
    agents[target].run.assert_called_once_with(task=task)
    for name, agent in agents.items():
        if name != target:
            agent.run.assert_not_called()
    namespace["swarm"].conversation.add.assert_called_with(
        role=target, content="Offline advice"
    )


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        '{"response_to_user":"Hi","agent_name":"Buddy-Agent"}',
    ],
)
def test_invalid_router_json_does_not_dispatch(
    example: tuple, response: str
) -> None:
    run_example, agents = example
    with pytest.raises(ValidationError):
        run_example(response)
    for agent in agents.values():
        agent.run.assert_not_called()
