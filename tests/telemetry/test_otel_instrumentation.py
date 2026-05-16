import importlib.util
from pathlib import Path


def load_otel_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "swarms"
        / "telemetry"
        / "otel.py"
    )
    spec = importlib.util.spec_from_file_location(
        "swarms_otel_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeSpanContext:
    def __init__(self, tracer, name, attributes):
        self.tracer = tracer
        self.name = name
        self.attributes = attributes

    def __enter__(self):
        self.tracer.spans.append((self.name, self.attributes))
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeTracer:
    def __init__(self):
        self.spans = []

    def start_as_current_span(self, name, attributes=None):
        return FakeSpanContext(self, name, attributes)


class ExampleAgent:
    agent_name = "researcher"
    id = "agent-1"


class ExampleWorkflow:
    name = "workflow"
    id = "workflow-1"
    agents = [ExampleAgent(), ExampleAgent()]


def test_otel_is_disabled_by_default(monkeypatch):
    otel = load_otel_module()
    monkeypatch.delenv("SWARMS_OTEL_ENABLED", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    assert otel.otel_enabled() is False

    def run(agent, task):
        return f"ran {task}"

    wrapped = otel.traced_method("swarms.agent.run")(run)
    assert wrapped(ExampleAgent(), "private task text") == (
        "ran private task text"
    )


def test_traced_method_records_safe_metadata_only(monkeypatch):
    otel = load_otel_module()
    tracer = FakeTracer()
    monkeypatch.setenv("SWARMS_OTEL_ENABLED", "true")
    monkeypatch.setattr(otel, "_TRACER", tracer)

    def run(workflow, task, prompt=None):
        assert task == "secret customer problem"
        assert prompt == "do not export me"
        return "ok"

    wrapped = otel.traced_method(
        "swarms.workflow.run",
        component_type="workflow",
    )(run)

    assert wrapped(
        ExampleWorkflow(),
        "secret customer problem",
        prompt="do not export me",
    ) == "ok"

    assert tracer.spans == [
        (
            "swarms.workflow.run",
            {
                "swarms.component_class": "ExampleWorkflow",
                "swarms.component_id": "workflow-1",
                "swarms.component_name": "workflow",
                "swarms.component_type": "workflow",
                "swarms.agent_count": 2,
            },
        )
    ]
    assert "secret customer problem" not in repr(tracer.spans)
    assert "do not export me" not in repr(tracer.spans)
