import functools
import inspect
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional


_TRACER = None
_SETUP_ATTEMPTED = False


def _otel_enabled() -> bool:
    value = os.getenv("SWARMS_OTEL_ENABLED", "").lower()
    return value in {"1", "true", "yes", "on"} or bool(
        os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    )


def _safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if isinstance(value, dict):
        return len(value)
    return str(value)


def _object_attributes(obj: Any) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {
        "swarms.component": obj.__class__.__name__,
    }

    for source_attr, otel_attr in (
        ("id", "swarms.id"),
        ("name", "swarms.name"),
        ("agent_name", "swarms.agent_name"),
        ("swarm_type", "swarms.swarm_type"),
        ("output_type", "swarms.output_type"),
    ):
        value = getattr(obj, source_attr, None)
        if value is not None:
            attrs[otel_attr] = _safe_value(value)

    agents = getattr(obj, "agents", None)
    if agents is not None:
        try:
            attrs["swarms.agents.count"] = len(agents)
        except TypeError:
            pass

    return attrs


def setup_otel() -> Optional[Any]:
    global _TRACER, _SETUP_ATTEMPTED

    if _TRACER is not None:
        return _TRACER
    if _SETUP_ATTEMPTED or not _otel_enabled():
        return None

    _SETUP_ATTEMPTED = True

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except Exception:
        return None

    resource = Resource.create(
        {
            "service.name": os.getenv(
                "SWARMS_OTEL_SERVICE_NAME", "swarms"
            )
        }
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            )
        )
    )
    trace.set_tracer_provider(provider)
    _TRACER = trace.get_tracer("swarms")
    return _TRACER


@contextmanager
def otel_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    tracer = setup_otel()
    if tracer is None:
        yield None
        return

    try:
        from opentelemetry.trace import Status, StatusCode
    except Exception:
        yield None
        return

    safe_attrs = {
        key: _safe_value(value)
        for key, value in (attributes or {}).items()
        if value is not None
    }

    with tracer.start_as_current_span(name) as span:
        for key, value in safe_attrs.items():
            span.set_attribute(key, value)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def trace_method(
    span_name: str,
    attributes_factory: Optional[
        Callable[[Any, tuple, dict], Dict[str, Any]]
    ] = None,
):
    def decorator(func):
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(self, *args, **kwargs):
                attributes = _object_attributes(self)
                if attributes_factory is not None:
                    attributes.update(
                        attributes_factory(self, args, kwargs)
                    )
                with otel_span(span_name, attributes):
                    return await func(self, *args, **kwargs)

            return async_wrapper

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            attributes = _object_attributes(self)
            if attributes_factory is not None:
                attributes.update(attributes_factory(self, args, kwargs))
            with otel_span(span_name, attributes):
                return func(self, *args, **kwargs)

        return wrapper

    return decorator
