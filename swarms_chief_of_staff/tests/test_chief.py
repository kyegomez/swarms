import pytest
from swarms_chief_of_staff.manager import ChiefOfStaff


class DummyAgent:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, prompt):
        return '{"goals": ["research"], "subtasks": [{"id": "1", "title": "do", "description": "desc"}], "suggestions": "use web"}'


def test_create_and_plan(monkeypatch):
    cs = ChiefOfStaff()
    t = cs.create_task("T1", "Research the market")
    import swarms_chief_of_staff.manager as mgr

    monkeypatch.setattr(mgr, "Agent", DummyAgent)
    parsed = cs.plan_task(t.id)
    assert "subtasks" in parsed
    assert t.status == "planned"


def test_approve():
    cs = ChiefOfStaff()
    t = cs.create_task("T2", "Do something")
    t.status = "planned"
    cs.approve_plan(t.id)
    assert t.status == "approved"
