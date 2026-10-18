"""Check cache demos against the real Conversation cache and counters."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples/utils/conversation_cache_examples"


@pytest.fixture
def agents(monkeypatch: pytest.MonkeyPatch) -> list:
    monkeypatch.setitem(
        sys.modules, "loguru", SimpleNamespace(logger=Mock())
    )
    monkeypatch.setitem(
        sys.modules,
        "swarms.utils.generate_id",
        SimpleNamespace(generate_id=lambda: "offline-id"),
    )
    monkeypatch.setitem(
        sys.modules,
        "swarms.utils.litellm_tokenizer",
        SimpleNamespace(
            count_tokens=lambda text, *args: len(text.split())
        ),
    )
    conversation = runpy.run_path(
        str(ROOT / "swarms/structs/conversation.py")
    )["Conversation"]
    created = []

    class OfflineAgent:
        def __init__(self, agent_name: str, **kwargs) -> None:
            self.agent_name = agent_name
            self.short_memory = conversation(
                name=agent_name, dynamic_context_window=False
            )
            created.append(self)

        def run(self, task: str) -> str:
            self.short_memory.add("user", task)
            self.short_memory.return_history_as_string()
            self.short_memory.add("assistant", "Offline answer")
            return "Offline answer"

    class OfflineWorkflow:
        def __init__(self, agents: list, **kwargs) -> None:
            self.agents = agents

        def run(self, task: str) -> None:
            for agent in self.agents:
                task = agent.run(task)

    monkeypatch.setitem(
        sys.modules, "swarms", SimpleNamespace(Agent=OfflineAgent)
    )
    monkeypatch.setitem(
        sys.modules,
        "swarms.structs.sequential_workflow",
        SimpleNamespace(SequentialWorkflow=OfflineWorkflow),
    )
    blocked = Mock(side_effect=AssertionError("No network"))
    monkeypatch.setattr("socket.socket", blocked)
    monkeypatch.setattr("socket.create_connection", blocked)
    return created


def test_sequential_demo_records_stable_history_hits(
    agents: list, capsys: pytest.CaptureFixture
) -> None:
    runpy.run_path(
        str(EXAMPLES / "sequential_cache_test.py"),
        run_name="__main__",
    )
    assert len(agents) == 3
    for agent in agents:
        stats = agent.short_memory.get_cache_stats()
        assert stats["hits"] == 3
        assert stats["misses"] == 2
        assert stats["cached_tokens"] > 0
    assert "Hits:         9" in capsys.readouterr().out


def test_interactive_demo_counts_reads_and_invalidates_after_add(
    agents: list,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(
        "builtins.input", Mock(side_effect=["Hello", "quit"])
    )
    runpy.run_path(
        str(EXAMPLES / "conversation_cache_interactive.py"),
        run_name="__main__",
    )
    memory = agents[0].short_memory
    assert memory.get_cache_stats()["misses"] == 1
    assert "Total calls: 1" in capsys.readouterr().out
    first = memory.return_history_as_string()
    assert memory.return_history_as_string() == first
    memory.add("user", "A new message")
    assert "A new message" in memory.return_history_as_string()
    assert memory.get_cache_stats()["hits"] == 1
    assert memory.get_cache_stats()["misses"] == 3
