import importlib.util
from pathlib import Path


def load_otel_module():
    path = (
        Path(__file__).parents[2]
        / "swarms"
        / "telemetry"
        / "otel.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_swarms_otel", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trace_method_is_noop_when_disabled(monkeypatch):
    otel = load_otel_module()
    monkeypatch.delenv("SWARMS_OTEL_ENABLED", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    class Dummy:
        name = "example"

        @otel.trace_method("dummy.run")
        def run(self, value):
            return value + 1

    assert Dummy().run(1) == 2


def test_object_attributes_exclude_task_content():
    otel = load_otel_module()

    class Dummy:
        id = "abc"
        name = "workflow"
        agent_name = "agent"
        output_type = "dict"
        agents = [object(), object()]

    attrs = otel._object_attributes(Dummy())

    assert attrs["swarms.component"] == "Dummy"
    assert attrs["swarms.id"] == "abc"
    assert attrs["swarms.name"] == "workflow"
    assert attrs["swarms.agent_name"] == "agent"
    assert attrs["swarms.output_type"] == "dict"
    assert attrs["swarms.agents.count"] == 2
    assert "task" not in attrs
