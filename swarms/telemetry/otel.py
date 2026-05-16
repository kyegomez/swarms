import os
from contextlib import nullcontext
from functools import wraps
from typing import Any, Callable, Dict, Optional


_TRACER = None
_CONFIGURED = False
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _truthy_env(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in _TRUE_VALUES


def otel_enabled() -> bool:
    """Return whether OpenTelemetry tracing should be active."""
    return _truthy_env("SWARMS_OTEL_ENABLED") or bool(
        os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    )


def _safe_attribute_value(value: Any) -> Optional[Any]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value[:128]
    return None


def span_attributes(
    component: Any,
    component_type: str,
) -> Dict[str, Any]:
    """Build safe metadata-only span attributes for a component.

    This deliberately avoids task, prompt, message, tool argument, and
    response content. Spans should identify the component that ran, not
    export user data.
    """
    attributes: Dict[str, Any] = {
        "swarms.component_class": component.__class__.__name__,
    }

    component_id = _safe_attribute_value(
        getattr(component, "id", None)
    )
    if component_id is not None:
        attributes["swarms.component_id"] = component_id

    component_name = _safe_attribute_value(
        getattr(component, "agent_name", None)
        or getattr(component, "name", None)
    )
    if component_name is not None:
        attributes["swarms.component_name"] = component_name

    attributes["swarms.component_type"] = component_type

    agents = getattr(component, "agents", None)
    if isinstance(agents, (list, tuple)):
        attributes["swarms.agent_count"] = len(agents)

    return attributes


def _configure_provider():
    global _CONFIGURED

    if _CONFIGURED:
        return

    _CONFIGURED = True
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()

    if not endpoint:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create(
            {
                "service.name": os.getenv(
                    "OTEL_SERVICE_NAME", "swarms"
                )
            }
        )
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        )
        trace.set_tracer_provider(provider)
    except Exception:
        return


def _get_tracer():
    global _TRACER

    if _TRACER is not None:
        return _TRACER

    if not otel_enabled():
        return None

    try:
        _configure_provider()
        from opentelemetry import trace

        _TRACER = trace.get_tracer("swarms")
        return _TRACER
    except Exception:
        return None


def start_span(span_name: str, attributes: Dict[str, Any]):
    if not otel_enabled():
        return nullcontext()

    tracer = _get_tracer()
    if tracer is None:
        return nullcontext()

    try:
        return tracer.start_as_current_span(
            span_name,
            attributes=attributes,
        )
    except Exception:
        return nullcontext()


def traced_method(
    span_name: str,
    component_type: str = "component",
) -> Callable:
    """Trace a method with safe component metadata when OTEL is enabled."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with start_span(
                span_name,
                span_attributes(self, component_type),
            ):
                return func(self, *args, **kwargs)

        return wrapper

    return decorator
