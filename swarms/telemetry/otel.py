import inspect
import importlib
import logging
import os
from contextlib import contextmanager, nullcontext
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, Optional


TRUE_VALUES = {"1", "true", "yes", "on"}
logger = logging.getLogger(__name__)


def _env_enabled(value: Optional[str]) -> bool:
    return bool(value and value.strip().lower() in TRUE_VALUES)


def is_opentelemetry_enabled() -> bool:
    return _env_enabled(os.getenv("SWARMS_OTEL_ENABLED")) or bool(
        os.getenv("SWARMS_OTEL_EXPORTER_OTLP_ENDPOINT")
    )


def _get_tracer() -> Any:
    if not is_opentelemetry_enabled():
        return None

    return _build_tracer(
        os.getenv("SWARMS_OTEL_EXPORTER_OTLP_ENDPOINT"),
        os.getenv("SWARMS_OTEL_SERVICE_NAME", "swarms"),
    )


@lru_cache(maxsize=8)
def _build_tracer(endpoint: Optional[str], service_name: str) -> Any:

    try:
        from opentelemetry import trace
    except ImportError:
        logger.warning(
            "OpenTelemetry tracing is enabled, but opentelemetry "
            "is not installed. Install opentelemetry-api, "
            "opentelemetry-sdk, and opentelemetry-exporter-otlp "
            "to export traces."
        )
        return None

    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        trace_exporter = importlib.import_module(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter"
        )
        provider = TracerProvider(
            resource=Resource.create(
                {"service.name": service_name}
            )
        )
        exporter_cls = trace_exporter.OTLPSpanExporter
        if endpoint:
            exporter = exporter_cls(endpoint=endpoint)
        else:
            exporter = exporter_cls()
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
    except Exception as exc:
        logger.warning(
            f"OpenTelemetry SDK exporter setup failed: {exc}. "
            "Falling back to the active tracer provider."
        )

    return trace.get_tracer("swarms")


@contextmanager
def opentelemetry_span(
    name: str, attributes: Optional[Dict[str, Any]] = None
):
    tracer = _get_tracer()
    if tracer is None:
        with nullcontext() as span:
            yield span
        return

    with tracer.start_as_current_span(name) as span:
        for key, value in (attributes or {}).items():
            if value is not None:
                span.set_attribute(key, value)
        yield span


def _safe_instance_name(instance: Any) -> Optional[str]:
    if instance is None:
        return None
    for attr in ("agent_name", "name", "id"):
        value = getattr(instance, attr, None)
        if value:
            return str(value)
    return instance.__class__.__name__


def trace_function(
    span_name: Optional[str] = None,
    component: Optional[str] = None,
) -> Callable:
    def decorator(func: Callable) -> Callable:
        name = span_name or f"{func.__module__}.{func.__name__}"

        def build_attributes(args: tuple) -> Dict[str, Any]:
            instance = args[0] if args else None
            return {
                "swarms.component": component,
                "swarms.instance_name": _safe_instance_name(instance),
                "swarms.function": func.__name__,
            }

        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with opentelemetry_span(
                    name, build_attributes(args)
                ) as span:
                    try:
                        result = await func(*args, **kwargs)
                        if span:
                            span.set_attribute("swarms.status", "ok")
                        return result
                    except Exception as exc:
                        if span:
                            span.record_exception(exc)
                            span.set_attribute(
                                "swarms.status", "error"
                            )
                        raise

            return async_wrapper

        @wraps(func)
        def wrapper(*args, **kwargs):
            attributes = build_attributes(args)
            with opentelemetry_span(name, attributes) as span:
                try:
                    result = func(*args, **kwargs)
                    if span:
                        span.set_attribute("swarms.status", "ok")
                    return result
                except Exception as exc:
                    if span:
                        span.record_exception(exc)
                        span.set_attribute("swarms.status", "error")
                    raise

        return wrapper

    return decorator
