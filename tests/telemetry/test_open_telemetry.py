from swarms.telemetry.open_telemetry import (
    build_method_attributes,
    open_telemetry_enabled,
    trace_method,
    trace_span,
)


class DummyAgent:
    id = "agent-1"
    agent_name = "researcher"
    model_name = "gpt-test"


def test_open_telemetry_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SWARMS_OTEL_ENABLED", raising=False)
    monkeypatch.delenv(
        "SWARMS_OPEN_TELEMETRY_ENABLED", raising=False
    )

    assert open_telemetry_enabled() is False

    with trace_span("swarms.test.disabled") as span:
        assert span is None


def test_trace_method_preserves_return_value(monkeypatch):
    monkeypatch.delenv("SWARMS_OTEL_ENABLED", raising=False)

    @trace_method("swarms.test.method")
    def run(agent, task):
        return f"{agent.agent_name}:{task}"

    assert run(DummyAgent(), "hello") == "researcher:hello"


def test_build_method_attributes_includes_agent_metadata():
    attrs = build_method_attributes(
        (DummyAgent(), "do work"),
        {},
    )

    assert attrs["swarms.component.class"] == "DummyAgent"
    assert attrs["swarms.component.id"] == "agent-1"
    assert attrs["swarms.component.name"] == "researcher"
    assert attrs["swarms.agent.model"] == "gpt-test"
    assert attrs["swarms.task.length"] == len("do work")
