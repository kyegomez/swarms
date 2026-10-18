"""Read actual temporary documents through the legal example."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples/guides/demos/legal/legal_swarm.py"
)


@pytest.fixture
def legal_swarm(monkeypatch: pytest.MonkeyPatch) -> type:
    modules = {
        "PyPDF2": SimpleNamespace(PdfReader=Mock()),
        "swarms.structs.agent": SimpleNamespace(Agent=object),
        "swarms.structs.conversation": SimpleNamespace(
            Conversation=Mock
        ),
        "swarms.structs.ma_blocks": SimpleNamespace(
            find_agent_by_name=Mock()
        ),
        "swarms.structs.ma_utils": SimpleNamespace(
            set_random_models_for_agents=lambda agents: agents
        ),
        "swarms.utils.history_output_formatter": SimpleNamespace(
            history_output_formatter=Mock()
        ),
        "swarms.utils.generate_keys": SimpleNamespace(
            generate_api_key=Mock()
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    blocked = Mock(side_effect=AssertionError("No network"))
    monkeypatch.setattr("socket.socket", blocked)
    monkeypatch.setattr("socket.create_connection", blocked)
    cls = runpy.run_path(str(EXAMPLE))["LegalSwarm"]
    monkeypatch.setattr(cls, "_initialize_agents", lambda self: [])
    return cls


def test_constructor_reads_document_list(
    legal_swarm: type, tmp_path: Path
) -> None:
    document = tmp_path / "terms.txt"
    document.write_text("合同条款 — café", encoding="utf-8")
    swarm = legal_swarm(documents=[str(document)])
    content = swarm.conversation.add.call_args.kwargs["content"]
    assert "Documents: 合同条款 — café" in content


def test_document_order_is_preserved(
    legal_swarm: type, tmp_path: Path
) -> None:
    documents = [tmp_path / "second.txt", tmp_path / "first.txt"]
    for path, text in zip(documents, ["Second", "First"]):
        path.write_text(text, encoding="utf-8")
    swarm = legal_swarm()
    assert swarm.handle_documents([str(p) for p in documents]) == (
        "Second\n\nFirst"
    )


def test_repeated_list_reads_current_file_contents(
    legal_swarm: type, tmp_path: Path
) -> None:
    document = tmp_path / "terms.txt"
    document.write_text("Original", encoding="utf-8")
    documents = [str(document)]
    swarm = legal_swarm()
    assert swarm.handle_documents(documents) == "Original"
    document.write_text("Amended", encoding="utf-8")
    assert swarm.handle_documents(documents) == "Amended"


def test_missing_document_returns_existing_error(
    legal_swarm: type, tmp_path: Path
) -> None:
    missing = tmp_path / "missing.txt"
    assert legal_swarm().handle_documents([str(missing)]) == (
        f"Error: File not found - {missing}"
    )


def test_constructor_without_documents_still_works(
    legal_swarm: type,
) -> None:
    swarm = legal_swarm()
    content = swarm.conversation.add.call_args.kwargs["content"]
    assert "Documents: None" in content
