import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import Mock

import pytest


@pytest.fixture
def todos(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Load the actual shared todo store without constructing any agents."""
    modules = {
        name: ModuleType(name)
        for name in (
            "loguru",
            "swarms",
            "swarms.structs",
            "swarms.structs.agent",
            "swarms.structs.conversation",
            "swarms.utils",
            "swarms.utils.history_output_formatter",
            "swarms.utils.output_types",
        )
    }
    modules["loguru"].logger = Mock()
    modules["swarms.structs.agent"].Agent = Mock()
    modules["swarms.structs.conversation"].Conversation = Mock()
    modules[
        "swarms.utils.history_output_formatter"
    ].history_output_formatter = Mock()
    modules["swarms.utils.output_types"].OutputType = str
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/guides/fairy_swarm/fairy_swarm.py"
    )
    return runpy.run_path(str(path))["SharedTodoList"]()


def test_add_history_keeps_original_status(todos: Any) -> None:
    """Later state changes must not rewrite an earlier add event."""
    todos.add_todo("one", "Task", "Details", "worker")
    todos.update_status("one", "completed", result="done")
    todos.request_revision("one", "Revise", "reviewer")

    original = todos.history[0]["details"]
    assert original["status"] == "pending"
    assert original["completed_at"] is None
    assert original["result"] is None
    assert original["needs_revision"] is False
    assert todos.todos["one"]["needs_revision"] is True
    assert [event["version"] for event in todos.history] == [1, 2, 3]


def test_history_detaches_nested_input_data(todos: Any) -> None:
    """Record nested data as it existed at the event's timestamp."""
    hints = {"position": {"x": 10}}
    dependencies = ["setup"]
    todos.add_todo(
        "one",
        "Task",
        "Details",
        "worker",
        spatial_hints=hints,
        depends_on=dependencies,
    )

    hints["position"]["x"] = 99
    dependencies.append("later")

    original = todos.history[0]["details"]
    assert original["spatial_hints"] == {"position": {"x": 10}}
    assert original["depends_on"] == ["setup"]


def test_update_history_retains_recorded_status(todos: Any) -> None:
    """Existing update events retain their own status and order."""
    todos.add_todo("one", "Task", "Details", "worker")
    todos.update_status("one", "in_progress")
    todos.update_status("one", "completed", result="done")

    assert [event["action"] for event in todos.history] == [
        "add",
        "update",
        "update",
    ]
    assert todos.history[1]["details"] == {"status": "in_progress"}
    assert todos.history[2]["details"] == {"status": "completed"}
