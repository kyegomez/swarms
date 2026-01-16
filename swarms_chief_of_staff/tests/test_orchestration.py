from swarms_chief_of_staff.manager import ChiefOfStaff


class DummyAgent:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, prompt):
        return f"result for: {prompt}"


def test_execute_plan(monkeypatch, tmp_path):
    cs = ChiefOfStaff(db_path=str(tmp_path / "test.db"))
    t = cs.create_task("T1", "Research the market")
    # simulate planned result
    t.result = {
        "subtasks": [
            {"title": "find data", "description": "Search for datasets"},
            {"title": "summarize", "description": "Summarize findings"},
    t.status = "planned"
    cs.storage.update_task(t)

    import swarms_chief_of_staff.manager as mgr

    monkeypatch.setattr(mgr, "Agent", DummyAgent)
    out = cs.execute_plan(t.id)
    assert "outputs" in out
    assert len(out["outputs"]) == 2
    # ensure persistence
    loaded = cs.storage.load_task(t.id)
    assert loaded.status == "executed"
from swarms_chief_of_staff.manager import ChiefOfStaff


class DummyAgent:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, prompt):
        return f"result for: {prompt}"


def test_execute_plan(monkeypatch, tmp_path):
    cs = ChiefOfStaff(db_path=str(tmp_path / "test.db"))
    t = cs.create_task("T1", "Research the market")
    # simulate planned result
    t.result = {
        "goals": ["research"],
        "subtasks": [
            {"title": "find data", "description": "Search for datasets"},
            {"title": "summarize", "description": "Summarize findings"},
        ],
    }
    t.status = "planned"
    cs.storage.update_task(t)

    import swarms_chief_of_staff.manager as mgr

    monkeypatch.setattr(mgr, "Agent", DummyAgent)
    out = cs.execute_plan(t.id)
    assert "outputs" in out
    assert len(out["outputs"]) == 2
    # ensure persistence
    loaded = cs.storage.load_task(t.id)
    assert loaded.status == "executed"
*** End Patch
