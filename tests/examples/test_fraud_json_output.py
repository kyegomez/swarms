"""Exercise classifier JSON with real conversation output formatting."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/guides/demos/hackathon_feb16/fraud.py"
SPECIALIST_RESULT = {"is_suspicious": False, "reason": "Offline"}
FINAL_RESULT = {"final_suspicious": False, "details": []}


@pytest.fixture
def classifier(monkeypatch: pytest.MonkeyPatch) -> Any:
    blocked = Mock(
        side_effect=AssertionError("No network or tokenizer")
    )
    for name, module in {
        "yaml": SimpleNamespace(),
        "loguru": SimpleNamespace(logger=Mock()),
        "swarms.utils.litellm_tokenizer": SimpleNamespace(
            count_tokens=blocked
        ),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr("socket.socket", blocked)
    monkeypatch.setattr("socket.create_connection", blocked)
    for name in ["generate_id", "output_types"]:
        namespace = runpy.run_path(
            str(ROOT / f"swarms/utils/{name}.py")
        )
        monkeypatch.setitem(
            sys.modules,
            f"swarms.utils.{name}",
            SimpleNamespace(**namespace),
        )
    conversation = runpy.run_path(
        str(ROOT / "swarms/structs/conversation.py")
    )["Conversation"]
    formatter = runpy.run_path(
        str(ROOT / "swarms/utils/history_output_formatter.py")
    )["history_output_formatter"]

    class OfflineAgent:
        def __init__(
            self,
            agent_name: str,
            output_type: str,
            system_prompt: str,
            **kwargs: object,
        ) -> None:
            self.agent_name = agent_name
            self.output_type = output_type
            self.memory = conversation(
                system_prompt=system_prompt,
                dynamic_context_window=False,
            )
            self.tasks = []
            if agent_name == "Boss-Agent":
                instructions = {
                    "geolocation_data": "Location instructions",
                    "ip_data": "IP instructions",
                    "email_data": "Account instructions",
                }
                self.responses = iter(
                    [
                        json.dumps(instructions),
                        json.dumps(FINAL_RESULT),
                    ]
                )
            else:
                self.responses = iter([json.dumps(SPECIALIST_RESULT)])

        def run(self, task: str) -> str:
            self.tasks.append(task)
            self.memory.add("User", task)
            self.memory.add(self.agent_name, next(self.responses))
            return formatter(self.memory, type=self.output_type)

    monkeypatch.setitem(
        sys.modules, "swarms", SimpleNamespace(Agent=OfflineAgent)
    )
    return runpy.run_path(str(EXAMPLE))["FraudClassifier"]()


def test_classifier_parses_final_model_json(classifier: Any) -> None:
    assert classifier.classify_row({"declared_country": "US"}) == (
        FINAL_RESULT
    )
    assert classifier.geolocation_agent.tasks == [
        "Location instructions"
    ]
    assert classifier.ip_agent.tasks == ["IP instructions"]
    assert classifier.email_agent.tasks == ["Account instructions"]


@pytest.mark.parametrize(
    "agent_name", ["geolocation_agent", "ip_agent", "email_agent"]
)
def test_specialist_returns_json_answer_content(
    classifier: Any, agent_name: str
) -> None:
    response = getattr(classifier, agent_name).run("Offline input")
    assert json.loads(response) == SPECIALIST_RESULT


def test_invalid_instructions_use_existing_fallback(
    classifier: Any,
) -> None:
    classifier.boss_agent.responses = iter(
        ["Invalid JSON", json.dumps(FINAL_RESULT)]
    )
    assert classifier.classify_row({"declared_country": "US"}) == (
        FINAL_RESULT
    )
    assert (
        "declared_country: US"
        in classifier.geolocation_agent.tasks[0]
    )
