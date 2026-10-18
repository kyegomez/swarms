"""Render generated reports to verify their Markdown structure."""

import runpy
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from xml.etree import ElementTree

import pytest

markdown = pytest.importorskip("markdown")
EXAMPLE_DIR = (
    Path(__file__).resolve().parents[2]
    / "examples/guides/demos/medical"
)


@pytest.fixture(params=["medical_coder_agent.py", "ollama_demo.py"])
def report(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Callable:
    blocked = Mock(side_effect=AssertionError("No network or files"))
    monkeypatch.setitem(
        sys.modules,
        "swarms",
        SimpleNamespace(
            Agent=lambda **kwargs: SimpleNamespace(**kwargs),
            AgentRearrange=Mock(),
            create_file_in_folder=blocked,
        ),
    )
    monkeypatch.setattr("socket.socket", blocked)
    monkeypatch.setattr("socket.create_connection", blocked)
    return runpy.run_path(str(EXAMPLE_DIR / request.param))[
        "generate_coding_report"
    ]


def render(text: str) -> ElementTree.Element:
    return ElementTree.fromstring(
        f"<root>{markdown.markdown(text)}</root>"
    )


def test_report_sections_render_as_headings(report: Callable) -> None:
    document = render(report("Offline summary."))
    assert [node.text for node in document.findall("h2")] == [
        "Clinical Summary",
        "Coding Summary",
        "Documentation and Compliance Notes",
        "Recommendations",
    ]
    assert len(document.findall("h3")) == 4
    assert document.find("pre") is None


def test_template_notes_render_as_lists(report: Callable) -> None:
    document = render(report("Offline summary."))
    assert len(document.findall("ul/li")) == 6


def test_embedded_markdown_keeps_its_structure(
    report: Callable,
) -> None:
    diagnosis = (
        "**Offline summary**\n\n- Finding\n\n    literal sample"
    )
    output = report(diagnosis)
    assert diagnosis in output
    document = render(output)
    assert document.find(".//strong").text == "Offline summary"


def test_title_and_diagnosis_text_are_preserved(
    report: Callable,
) -> None:
    output = report("Offline summary.")
    assert output.startswith(
        "# Medical Diagnosis and Coding Report\n"
    )
    assert "Offline summary." in output
