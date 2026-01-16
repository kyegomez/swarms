import pytest
from taskmaster.manager import TaskManager


class DummyAgent:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, prompt):
        return '{"goals": ["research"], "subtasks": [{"id": "1", "title": "do", "description": "desc"}], "suggestions": "use web"}'


def test_create_and_plan(monkeypatch):
    tm = TaskManager()
    t = tm.create_task("T1", "Research the market")
    # patch Agent to avoid external calls
    import taskmaster.manager as mgr

    monkeypatch.setattr(mgr, "Agent", DummyAgent)
    parsed = tm.plan_task(t.id)
    assert "subtasks" in parsed
    assert t.status == "planned"


def test_approve():
    tm = TaskManager()
    t = tm.create_task("T2", "Do something")
    t.status = "planned"
    tm.approve_plan(t.id)
    assert t.status == "approved"
