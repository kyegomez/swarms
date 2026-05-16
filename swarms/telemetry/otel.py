"""
OpenTelemetry Integration for Swarms

This module provides optional OpenTelemetry tracing and metrics for
agent executions and multi-agent workflows. Enable by setting the
SWARMS_OTEL_ENABLED environment variable to 'true'.

Configuration via environment variables:
    SWARMS_OTEL_ENABLED: Set to 'true' to enable tracing (default: false)
    OTEL_SERVICE_NAME: Service name for traces (default: 'swarms')
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint URL (optional)
    OTEL_EXPORTER_OTLP_HEADERS: OTLP headers (optional)
"""

import os
import time
import functools
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    TypeVar,
    Union,
)
from contextlib import contextmanager

F = TypeVar("F", bound=Callable[..., Any])

_OTEL_AVAILABLE = False
_tracer = None
_meter = None

try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        PeriodicExportingMetricReader,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.trace import Status, StatusCode, Span
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )

    _OTEL_AVAILABLE = True
except ImportError:
    pass


def _is_otel_enabled() -> bool:
    """Check if OpenTelemetry is enabled via environment variable."""
    enabled = os.getenv("SWARMS_OTEL_ENABLED", "false").lower()
    return enabled in ("true", "1", "yes") and _OTEL_AVAILABLE


def _get_service_name() -> str:
    """Get the service name from environment or default."""
    return os.getenv("OTEL_SERVICE_NAME", "swarms")


def _init_tracer():
    """Initialize and return the OpenTelemetry tracer."""
    global _tracer
    if _tracer is not None:
        return _tracer

    if not _is_otel_enabled():
        return None

    service_name = _get_service_name()
    resource = Resource.create(
        {ResourceAttributes.SERVICE_NAME: service_name}
    )

    provider = TracerProvider(resource=resource)

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        exporter = OTLPSpanExporter(endpoint=endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("swarms.telemetry", "1.0.0")
    return _tracer


def _init_meter():
    """Initialize and return the OpenTelemetry meter."""
    global _meter
    if _meter is not None:
        return _meter

    if not _is_otel_enabled():
        return None

    service_name = _get_service_name()
    resource = Resource.create(
        {ResourceAttributes.SERVICE_NAME: service_name}
    )

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        exporter = OTLPMetricExporter(endpoint=endpoint)
        reader = PeriodicExportingMetricReader(exporter)
        provider = MeterProvider(
            resource=resource, metric_readers=[reader]
        )
    else:
        provider = MeterProvider(resource=resource)

    metrics.set_meter_provider(provider)
    _meter = metrics.get_meter("swarms.telemetry", "1.0.0")
    return _meter


class OTelMetrics:
    """Container for OpenTelemetry metrics instruments."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.agent_runs = None
        self.agent_duration = None
        self.agent_errors = None
        self.swarm_runs = None
        self.swarm_duration = None

        if not _is_otel_enabled():
            return

        meter = _init_meter()
        if meter is None:
            return

        self.agent_runs = meter.create_counter(
            name="swarms.agent.runs",
            description="Number of agent run invocations",
            unit="1",
        )

        self.agent_duration = meter.create_histogram(
            name="swarms.agent.duration",
            description="Duration of agent runs in milliseconds",
            unit="ms",
        )

        self.agent_errors = meter.create_counter(
            name="swarms.agent.errors",
            description="Number of agent run errors",
            unit="1",
        )

        self.swarm_runs = meter.create_counter(
            name="swarms.swarm.runs",
            description="Number of swarm run invocations",
            unit="1",
        )

        self.swarm_duration = meter.create_histogram(
            name="swarms.swarm.duration",
            description="Duration of swarm runs in milliseconds",
            unit="ms",
        )


def get_tracer():
    """Get the initialized tracer, or None if OTEL is disabled."""
    return _init_tracer()


def get_metrics() -> OTelMetrics:
    """Get the metrics container instance."""
    return OTelMetrics()


@contextmanager
def _safe_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind=None,
):
    """Context manager that yields a span or no-op context."""
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    if kind is None and _OTEL_AVAILABLE:
        kind = trace.SpanKind.INTERNAL

    with tracer.start_as_current_span(
        name, kind=kind, attributes=attributes or {}
    ) as span:
        yield span


def _safe_set_attribute(
    span: Optional[Any], key: str, value: Any
) -> None:
    """Safely set a span attribute if span exists."""
    if span is None:
        return
    try:
        if value is not None:
            if isinstance(value, (str, int, float, bool)):
                span.set_attribute(key, value)
            elif isinstance(value, (list, tuple)):
                if all(
                    isinstance(v, (str, int, float, bool))
                    for v in value
                ):
                    span.set_attribute(key, list(value))
    except Exception:
        pass


def _safe_set_status(
    span: Optional[Any], success: bool, description: str = ""
) -> None:
    """Safely set span status if span exists."""
    if span is None or not _OTEL_AVAILABLE:
        return
    try:
        if success:
            span.set_status(Status(StatusCode.OK))
        else:
            span.set_status(Status(StatusCode.ERROR, description))
    except Exception:
        pass


def _safe_record_exception(
    span: Optional[Any], exception: Exception
) -> None:
    """Safely record an exception on the span."""
    if span is None:
        return
    try:
        span.record_exception(exception)
    except Exception:
        pass


def trace_agent_run(func: F) -> F:
    """
    Decorator to trace agent run method execution.

    Creates a span for each agent.run() call with relevant metadata.
    Only activates if SWARMS_OTEL_ENABLED=true and dependencies are
    available.

    Attributes recorded:
        - agent.id: Agent's unique identifier
        - agent.name: Agent's name
        - agent.model: Model name used by the agent
        - agent.max_loops: Maximum loops configured
        - run.has_image: Whether image input was provided
        - run.status: 'success' or 'error'
        - run.duration_ms: Execution time in milliseconds
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not _is_otel_enabled():
            return func(self, *args, **kwargs)

        agent_name = getattr(self, "agent_name", "unknown")
        agent_id = getattr(self, "id", None)
        model_name = getattr(self, "model_name", None)
        max_loops = getattr(self, "max_loops", None)

        task = kwargs.get("task") or (args[0] if args else None)
        has_image = kwargs.get("img") is not None or kwargs.get(
            "imgs"
        ) is not None

        span_attrs = {
            "agent.name": agent_name,
            "run.has_image": has_image,
        }
        if agent_id:
            span_attrs["agent.id"] = str(agent_id)
        if model_name:
            span_attrs["agent.model"] = model_name
        if max_loops:
            span_attrs["agent.max_loops"] = max_loops

        metrics = get_metrics()
        start_time = time.perf_counter()

        with _safe_span(
            f"agent.run.{agent_name}", attributes=span_attrs
        ) as span:
            try:
                result = func(self, *args, **kwargs)

                duration_ms = (
                    time.perf_counter() - start_time
                ) * 1000
                _safe_set_attribute(
                    span, "run.duration_ms", duration_ms
                )
                _safe_set_attribute(span, "run.status", "success")
                _safe_set_status(span, True)

                if metrics.agent_runs:
                    metrics.agent_runs.add(
                        1, {"agent.name": agent_name, "status": "success"}
                    )
                if metrics.agent_duration:
                    metrics.agent_duration.record(
                        duration_ms, {"agent.name": agent_name}
                    )

                return result

            except Exception as e:
                duration_ms = (
                    time.perf_counter() - start_time
                ) * 1000
                _safe_set_attribute(
                    span, "run.duration_ms", duration_ms
                )
                _safe_set_attribute(span, "run.status", "error")
                _safe_set_attribute(
                    span, "error.type", type(e).__name__
                )
                _safe_record_exception(span, e)
                _safe_set_status(span, False, str(e))

                if metrics.agent_runs:
                    metrics.agent_runs.add(
                        1, {"agent.name": agent_name, "status": "error"}
                    )
                if metrics.agent_errors:
                    metrics.agent_errors.add(
                        1,
                        {
                            "agent.name": agent_name,
                            "error.type": type(e).__name__,
                        },
                    )
                if metrics.agent_duration:
                    metrics.agent_duration.record(
                        duration_ms, {"agent.name": agent_name}
                    )

                raise

    return wrapper


def trace_swarm_run(func: F) -> F:
    """
    Decorator to trace swarm/workflow run method execution.

    Creates a span for multi-agent workflow executions with metadata
    about the swarm configuration and execution.

    Attributes recorded:
        - swarm.id: Swarm's unique identifier
        - swarm.name: Swarm's name
        - swarm.type: Type of swarm (e.g., SequentialWorkflow)
        - swarm.agent_count: Number of agents in the swarm
        - run.status: 'success' or 'error'
        - run.duration_ms: Execution time in milliseconds
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not _is_otel_enabled():
            return func(self, *args, **kwargs)

        swarm_name = getattr(self, "name", "unknown")
        swarm_id = getattr(self, "id", None)
        swarm_type = self.__class__.__name__
        agents = getattr(self, "agents", [])
        agent_count = len(agents) if agents else 0

        span_attrs = {
            "swarm.name": swarm_name,
            "swarm.type": swarm_type,
            "swarm.agent_count": agent_count,
        }
        if swarm_id:
            span_attrs["swarm.id"] = str(swarm_id)

        metrics = get_metrics()
        start_time = time.perf_counter()

        with _safe_span(
            f"swarm.run.{swarm_type}", attributes=span_attrs
        ) as span:
            try:
                result = func(self, *args, **kwargs)

                duration_ms = (
                    time.perf_counter() - start_time
                ) * 1000
                _safe_set_attribute(
                    span, "run.duration_ms", duration_ms
                )
                _safe_set_attribute(span, "run.status", "success")
                _safe_set_status(span, True)

                if metrics.swarm_runs:
                    metrics.swarm_runs.add(
                        1,
                        {"swarm.type": swarm_type, "status": "success"},
                    )
                if metrics.swarm_duration:
                    metrics.swarm_duration.record(
                        duration_ms, {"swarm.type": swarm_type}
                    )

                return result

            except Exception as e:
                duration_ms = (
                    time.perf_counter() - start_time
                ) * 1000
                _safe_set_attribute(
                    span, "run.duration_ms", duration_ms
                )
                _safe_set_attribute(span, "run.status", "error")
                _safe_set_attribute(
                    span, "error.type", type(e).__name__
                )
                _safe_record_exception(span, e)
                _safe_set_status(span, False, str(e))

                if metrics.swarm_runs:
                    metrics.swarm_runs.add(
                        1,
                        {"swarm.type": swarm_type, "status": "error"},
                    )

                raise

    return wrapper


@contextmanager
def trace_context(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
):
    """
    Context manager for custom span creation.

    Use this to trace custom code blocks within agent or swarm
    operations.

    Args:
        name: Name for the span
        attributes: Optional dict of span attributes

    Yields:
        The span object (or None if OTEL is disabled)

    Example:
        with trace_context("custom.operation", {"key": "value"}) as span:
            # your code here
            if span:
                span.set_attribute("result.count", 42)
    """
    with _safe_span(name, attributes=attributes) as span:
        yield span


def trace_tool_execution(
    tool_name: str, agent_name: str = "unknown"
) -> Callable[[F], F]:
    """
    Decorator factory for tracing tool executions.

    Args:
        tool_name: Name of the tool being executed
        agent_name: Name of the agent executing the tool

    Returns:
        Decorator function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not _is_otel_enabled():
                return func(*args, **kwargs)

            span_attrs = {
                "tool.name": tool_name,
                "agent.name": agent_name,
            }

            with _safe_span(
                f"tool.execute.{tool_name}", attributes=span_attrs
            ) as span:
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (
                        time.perf_counter() - start_time
                    ) * 1000
                    _safe_set_attribute(
                        span, "tool.duration_ms", duration_ms
                    )
                    _safe_set_attribute(span, "tool.status", "success")
                    _safe_set_status(span, True)
                    return result
                except Exception as e:
                    duration_ms = (
                        time.perf_counter() - start_time
                    ) * 1000
                    _safe_set_attribute(
                        span, "tool.duration_ms", duration_ms
                    )
                    _safe_set_attribute(span, "tool.status", "error")
                    _safe_record_exception(span, e)
                    _safe_set_status(span, False, str(e))
                    raise

        return wrapper

    return decorator


def is_otel_enabled() -> bool:
    """
    Check if OpenTelemetry tracing is currently enabled.

    Returns:
        True if OTEL is enabled and dependencies are available.
    """
    return _is_otel_enabled()


def otel_available() -> bool:
    """
    Check if OpenTelemetry dependencies are installed.

    Returns:
        True if opentelemetry packages are available.
    """
    return _OTEL_AVAILABLE
