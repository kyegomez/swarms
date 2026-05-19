import os
from contextlib import contextmanager
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, Optional

from loguru import logger


TRUE_VALUES = {"1", "true", "yes", "on"}


def open_telemetry_enabled() -> bool:
    """Return whether OpenTelemetry tracing is enabled."""
    value = os.getenv("SWARMS_OTEL_ENABLED") or os.getenv(
        "SWARMS_OPEN_TELEMETRY_ENABLED"
    )
    return str(value).lower() in TRUE_VALUES


def _safe_attribute_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _clean_attributes(
    attributes: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not attributes:
        return {}
    return {
        key: _safe_attribute_value(value)
        for key, value in attributes.items()
        if value is not None
    }


def _task_attributes(args: tuple, kwargs: dict) -> Dict[str, Any]:
    task = kwargs.get("task")
    if task is None and len(args) > 1:
        task = args[1]
    if task is None:
        return {}
    return {
        "swarms.task.length": len(str(task)),
        "swarms.task.type": type(task).__name__,
    }


def _object_attributes(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}

    attrs = {
        "swarms.component.class": obj.__class__.__name__,
        "swarms.component.id": getattr(obj, "id", None),
        "swarms.component.name": getattr(
            obj,
            "agent_name",
            getattr(obj, "name", None),
        ),
        "swarms.agent.model": getattr(obj, "model_name", None),
        "swarms.swarm.type": getattr(obj, "swarm_type", None),
        "swarms.agent.count": (
            len(getattr(obj, "agents", []))
            if getattr(obj, "agents", None) is not None
            else None
        ),
    }
    return _clean_attributes(attrs)


def build_method_attributes(
    args: tuple,
    kwargs: dict,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    attrs = {}
    if args:
        attrs.update(_object_attributes(args[0]))
    attrs.update(_task_attributes(args, kwargs))
    attrs.update(_clean_attributes(extra))
    return attrs


@lru_cache(maxsize=1)
def _otel_components():
    if not open_telemetry_enabled():
        return None

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
        from opentelemetry.trace import Status, StatusCode
    except ImportError as exc:
        logger.warning(
            "OpenTelemetry tracing is enabled but dependencies are "
            f"missing: {exc}"
        )
        return None

    provider = trace.get_tracer_provider()
    if provider.__class__.__name__ == "ProxyTracerProvider":
        resource = Resource.create(
            {
                "service.name": os.getenv(
                    "OTEL_SERVICE_NAME", "swarms"
                )
            }
        )
        provider = TracerProvider(resource=resource)

        exporter_name = os.getenv(
            "SWARMS_OTEL_EXPORTER", "otlp"
        ).lower()
        if exporter_name == "console":
            exporter = ConsoleSpanExporter()
        else:
            try:
                from opentelemetry.exporter.otlp.proto.http import (
                    trace_exporter,
                )

                exporter = trace_exporter.OTLPSpanExporter()
            except ImportError as exc:
                logger.warning(
                    "OTLP exporter is unavailable; falling back to "
                    f"console exporter: {exc}"
                )
                exporter = ConsoleSpanExporter()

        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

    return {
        "tracer": trace.get_tracer("swarms"),
        "Status": Status,
        "StatusCode": StatusCode,
    }


@contextmanager
def trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
):
    components = _otel_components()
    if components is None:
        yield None
        return

    tracer = components["tracer"]
    Status = components["Status"]
    StatusCode = components["StatusCode"]

    with tracer.start_as_current_span(
        name,
        attributes=_clean_attributes(attributes),
    ) as span:
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        else:
            span.set_status(Status(StatusCode.OK))


def trace_method(
    span_name: str,
    extra_attributes: Optional[Dict[str, Any]] = None,
) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            attrs = build_method_attributes(
                args,
                kwargs,
                extra_attributes,
            )
            with trace_span(span_name, attrs):
                return func(*args, **kwargs)

        return wrapper

    return decorator
