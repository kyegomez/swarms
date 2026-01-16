import importlib.util
import pathlib


def _load_department_manager():
    path = pathlib.Path(__file__).parent.parent / "swarms" / "structs" / "department_manager.py"
    spec = importlib.util.spec_from_file_location("department_manager", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_department_manager_add_and_flatten():
    mod = _load_department_manager()
    DepartmentManager = getattr(mod, "DepartmentManager")
    dm = DepartmentManager()
    assert dm.list_departments() == []

    # create two fake agents (simple objects with .agent_name)
    a1 = type("A", (), {"agent_name": "a1"})()
    a2 = type("A", (), {"agent_name": "a2"})()

    dm.add_department("dept1", [a1])
    dm.add_agent_to_department("dept1", a2)

    depts = dm.list_departments()
    assert "dept1" in depts

    flat = dm.flatten_agents()
    assert len(flat) == 2
    assert flat[0].agent_name == "a1"
    assert flat[1].agent_name == "a2"
