"""Verify chart questions reach the real reformulation prompt."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples/guides/demos/chart_swarm.py"
)


@pytest.fixture
def chart_citor(monkeypatch: pytest.MonkeyPatch) -> object:
    blocked = Mock(side_effect=AssertionError("No provider calls"))
    for name, module in {
        "litellm": SimpleNamespace(completion=blocked),
        "loguru": SimpleNamespace(logger=Mock()),
        "swarms": SimpleNamespace(Agent=blocked),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr("socket.socket", blocked)
    monkeypatch.setattr("socket.create_connection", blocked)
    cls = runpy.run_path(str(EXAMPLE))["ChartCitor"]
    citor = object.__new__(cls)
    for stage in [
        "extraction",
        "reformulation",
        "captioning",
        "prefilter",
        "rerank",
        "localization",
    ]:
        agent = SimpleNamespace(
            run=Mock(return_value=f"Offline {stage} output")
        )
        setattr(citor, f"{stage}_agent", agent)
    return citor


@pytest.mark.parametrize(
    "question",
    [
        "What is the highest volume day?",
        "哪个月份的收入为 203？",
    ],
)
def test_question_reaches_answer_reformulation(
    chart_citor: object, question: str
) -> None:
    chart_citor.run("chart.png", question, "203")
    prompt = chart_citor.reformulation_agent.run.call_args.args[0]
    assert question in prompt


def test_existing_answer_table_and_image_are_preserved(
    chart_citor: object,
) -> None:
    result = chart_citor.run("chart.png", "Which day?", "203")
    call = chart_citor.reformulation_agent.run.call_args
    assert "203" in call.args[0]
    assert "Offline extraction output" in call.args[0]
    assert call.kwargs == {"img": "chart.png"}
    assert "Offline reformulation output" in result
    assert "Offline rerank output" in result
    assert "Offline localization output" in result
