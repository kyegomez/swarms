import asyncio

from swarms.telemetry.otel import (
    is_opentelemetry_enabled,
    opentelemetry_span,
    trace_function,
)


def test_opentelemetry_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SWARMS_OTEL_ENABLED", raising=False)
    monkeypatch.delenv(
        "SWARMS_OTEL_EXPORTER_OTLP_ENDPOINT", raising=False
    )

    assert is_opentelemetry_enabled() is False
    with opentelemetry_span("test.disabled") as span:
        assert span is None


def test_trace_function_noops_when_otel_missing(monkeypatch):
    monkeypatch.setenv("SWARMS_OTEL_ENABLED", "true")

    class Worker:
        name = "unit-worker"

        @trace_function("test.worker.run", component="test")
        def run(self, value):
            return value + 1

    assert Worker().run(1) == 2


def test_trace_function_preserves_exceptions(monkeypatch):
    monkeypatch.setenv("SWARMS_OTEL_ENABLED", "true")

    class Worker:
        agent_name = "failing-worker"

        @trace_function("test.worker.fail", component="test")
        def run(self):
            raise RuntimeError("boom")

    try:
        Worker().run()
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("RuntimeError was not raised")


def test_trace_function_supports_async_functions(monkeypatch):
    monkeypatch.setenv("SWARMS_OTEL_ENABLED", "true")

    class Worker:
        id = "async-worker"

        @trace_function("test.worker.async", component="test")
        async def run(self, value):
            return value * 2

    assert asyncio.run(Worker().run(3)) == 6
