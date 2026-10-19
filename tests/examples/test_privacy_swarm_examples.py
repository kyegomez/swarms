import runpy
from pathlib import Path
from typing import Any

import pytest

import swarms
from swarms.structs.conversation import Conversation


@pytest.mark.parametrize(
    "example,names",
    [
        (
            "privacy_building.py",
            [
                "MatchmakerAgent",
                "ProfileAnalyzer",
                "ConnectionFacilitator",
            ],
        ),
        (
            "medical/health_privacy_swarm.py",
            [
                "HealthScoreGatekeeper",
                "BossAgent",
                "HealthScoreAnalyzer",
                "ReportGenerator",
            ],
        ),
        (
            "medical/health_privacy_swarm_two.py",
            [
                "PatientAgent",
                "DoctorAgent",
                "NurseAgent",
                "MedicalRecordsAgent",
            ],
        ),
    ],
)
def test_privacy_example_routes_to_final_answer(
    example: str,
    names: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Execute each full demo with real routing and deterministic agents."""
    calls = []

    class ScriptAgent:
        def __init__(self, agent_name: str, **kwargs: Any) -> None:
            self.agent_name = agent_name
            self.short_memory = Conversation()

        def run(
            self, task: str, messages: list[dict], **kwargs: Any
        ) -> str:
            calls.append((self.agent_name, task, messages))
            self.short_memory.add(
                self.agent_name, f"{self.agent_name} answer"
            )
            return "Full agent transcript, not the final answer"

    monkeypatch.setattr(swarms, "Agent", ScriptAgent)
    monkeypatch.chdir(tmp_path)
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/guides/demos"
        / example
    )
    result = runpy.run_path(str(path), run_name="__main__")

    assert [name for name, _, _ in calls] == names
    prior = []
    for index, (_, task, messages) in enumerate(calls):
        expected_task = (
            result["task"]
            if index == 0
            else f"{names[index - 1]}: {names[index - 1]} answer"
        )
        assert task == expected_task
        assert [m for m in messages if m["role"] != "system"] == prior
        prior.append({"role": "user", "content": expected_task})
    assert result["output"] == f"{names[-1]} answer"
    assert capsys.readouterr().out.strip() == result["output"]
    assert [
        turn["role"]
        for turn in result[
            "agent_system"
        ].conversation.conversation_history
    ] == ["User", *names]
