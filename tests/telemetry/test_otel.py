import pytest

from swarms.telemetry.otel import (
    is_opentelemetry_enabled,
    method_span_attributes,
    task_attributes,
    trace_otel_method,
)


def test_opentelemetry_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SWARMS_OTEL_ENABLED", raising=False)
    monkeypatch.delenv("SWARMS_TELEMETRY_OTEL", raising=False)
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)

    assert is_opentelemetry_enabled() is False


def test_task_attributes_do_not_record_content_by_default(
    monkeypatch,
):
    monkeypatch.delenv("SWARMS_OTEL_RECORD_CONTENT", raising=False)

    attributes = task_attributes(task="private task", imgs=["a", "b"])

    assert attributes["swarms.task.length"] == len("private task")
    assert attributes["swarms.task.has_image"] is True
    assert attributes["swarms.task.image_count"] == 2
    assert "swarms.task.content" not in attributes


def test_task_attributes_can_record_content_when_enabled(monkeypatch):
    monkeypatch.setenv("SWARMS_OTEL_RECORD_CONTENT", "true")

    attributes = task_attributes(task="record me")

    assert attributes["swarms.task.content"] == "record me"


def test_method_span_attributes_maps_common_run_arguments():
    class DummySwarm:
        id = "swarm-id"
        name = "dummy"
        agents = [object(), object()]
        max_loops = 2
        output_type = "final"

        def run(self, task=None, img=None, imgs=None):
            return task

    swarm = DummySwarm()

    attributes = method_span_attributes(
        swarm,
        DummySwarm.run,
        ("hello",),
        {"imgs": ["one.png"]},
    )

    assert attributes["swarms.swarm.id"] == "swarm-id"
    assert attributes["swarms.swarm.name"] == "dummy"
    assert attributes["swarms.swarm.agent_count"] == 2
    assert attributes["swarms.task.length"] == len("hello")
    assert attributes["swarms.task.has_image"] is True
    assert attributes["swarms.task.image_count"] == 1
    assert attributes["swarms.method.name"] == "run"


def test_trace_otel_method_is_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("SWARMS_OTEL_ENABLED", raising=False)
    monkeypatch.delenv("SWARMS_TELEMETRY_OTEL", raising=False)

    class DummySwarm:
        id = "swarm-id"
        name = "dummy"
        agents = []

        @trace_otel_method("test.dummy.run")
        def run(self, task):
            return f"done: {task}"

    assert DummySwarm().run("work") == "done: work"


def test_trace_otel_method_preserves_exceptions(monkeypatch):
    monkeypatch.delenv("SWARMS_OTEL_ENABLED", raising=False)
    monkeypatch.delenv("SWARMS_TELEMETRY_OTEL", raising=False)

    class DummySwarm:
        agents = []

        @trace_otel_method("test.dummy.run")
        def run(self, task):
            raise ValueError(task)

    with pytest.raises(ValueError, match="boom"):
        DummySwarm().run("boom")
