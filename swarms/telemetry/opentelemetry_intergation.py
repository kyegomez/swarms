"""
OpenTelemetry integration for Swarms framework.

Provides distributed tracing, metrics, and logging capabilities across
agents and multi-agent structures using OpenTelemetry standards.

Configuration via environment variables:
    OTEL_SERVICE_NAME: Service name (default: "swarms")
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint URL
    OTEL_EXPORTER_OTLP_HEADERS: Headers for OTLP exporter (JSON format)
    OTEL_TRACES_EXPORTER: Traces exporter (default: "otlp")
    OTEL_METRICS_EXPORTER: Metrics exporter (default: "otlp")
    OTEL_LOGS_EXPORTER: Logs exporter (default: "otlp")
    OTEL_ENABLED: Enable/disable OpenTelemetry (default: "true")
    OTEL_SDK_DISABLED: Disable OpenTelemetry SDK (default: "false")
"""

import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional
from functools import wraps

from loguru import logger

_otel_available = False
_tracer = None
_meter = None
_logger = None

try:
    from opentelemetry import trace, metrics, _logs
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
        OTLPLogExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    _otel_available = True
except ImportError:
    pass


def _is_otel_enabled() -> bool:
    """Check if OpenTelemetry is enabled via environment variables."""
    if os.getenv("OTEL_SDK_DISABLED", "false").lower() == "true":
        return False
    return os.getenv("OTEL_ENABLED", "true").lower() == "true"


def _parse_headers(headers_str: str) -> Dict[str, str]:
    """Parse headers from JSON string or key=value format."""
    import json

    try:
        return json.loads(headers_str)
    except (json.JSONDecodeError, TypeError):
        headers = {}
        for pair in headers_str.split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                headers[key.strip()] = value.strip()
        return headers


def _initialize_otel():
    """Initialize OpenTelemetry SDK with configuration from environment variables."""
    global _tracer, _meter, _logger

    if not _otel_available or not _is_otel_enabled():
        return False

    try:
        service_name = os.getenv("OTEL_SERVICE_NAME", "swarms")
        resource = Resource.create({"service.name": service_name})

        trace_provider = TracerProvider(resource=resource)
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        headers = {}

        if otlp_endpoint:
            headers = _parse_headers(
                os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "{}")
            )
            span_exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                headers=headers,
            )
            trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))

        trace.set_tracer_provider(trace_provider)
        _tracer = trace.get_tracer(__name__)

        if otlp_endpoint:
            meter_provider = MeterProvider(
                resource=resource,
                metric_readers=[
                    PeriodicExportingMetricReader(
                        OTLPMetricExporter(
                            endpoint=otlp_endpoint,
                            headers=headers,
                        )
                    )
                ],
            )
            metrics.set_meter_provider(meter_provider)
            _meter = metrics.get_meter(__name__)

            logger_provider = LoggerProvider(resource=resource)
            log_exporter = OTLPLogExporter(
                endpoint=otlp_endpoint,
                headers=headers,
            )
            logger_provider.add_log_record_processor(
                BatchLogRecordProcessor(log_exporter)
            )
            _logs.set_logger_provider(logger_provider)
            _logger = _logs.get_logger(__name__)

        return True

    except Exception as e:
        logger.debug(f"Failed to initialize OpenTelemetry: {e}")
        return False


if _otel_available and _is_otel_enabled():
    _initialize_otel()


def get_tracer(name: Optional[str] = None):
    """Get OpenTelemetry tracer instance."""
    if not _otel_available or not _is_otel_enabled():
        return None
    return trace.get_tracer(name or __name__)


def get_meter(name: Optional[str] = None):
    """Get OpenTelemetry meter instance."""
    if not _otel_available or not _is_otel_enabled():
        return None
    return metrics.get_meter(name or __name__)


@contextmanager
def trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind: Optional[Any] = None,
):
    """
    Context manager for creating a trace span.

    Args:
        name: Span name
        attributes: Dictionary of span attributes
        kind: Span kind (INTERNAL, SERVER, CLIENT, etc.)
    """
    if not _otel_available or not _is_otel_enabled():
        yield None
        return

    tracer = get_tracer()
    if not tracer:
        yield None
        return

    span_kind = kind if kind is not None else trace.SpanKind.INTERNAL
    span = tracer.start_span(name=name, kind=span_kind)

    if attributes:
        for key, value in attributes.items():
            try:
                span.set_attribute(key, str(value))
            except Exception:
                pass

    try:
        with trace.use_span(span):
            yield span
    except Exception as e:
        try:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
        except Exception:
            pass
        raise
    finally:
        try:
            span.end()
        except Exception:
            pass


def trace_function(
    span_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    capture_args: bool = True,
):
    """
    Decorator to trace function execution.

    Args:
        span_name: Custom span name (defaults to function name)
        attributes: Additional attributes to add to span
        capture_args: Whether to capture function arguments as attributes
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not _otel_available or not _is_otel_enabled():
                return func(*args, **kwargs)

            name = span_name or f"{func.__module__}.{func.__name__}"
            span_attrs = (attributes or {}).copy()

            if capture_args:
                import inspect

                try:
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()

                    for param_name, param_value in bound.arguments.items():
                        if param_name != "self":
                            try:
                                span_attrs[
                                    f"function.{param_name}"
                                ] = str(param_value)[:200]
                            except Exception:
                                pass
                except Exception:
                    pass

            with trace_span(name, span_attrs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time

                    record_metric(
                        "function.execution.time",
                        execution_time,
                        {"function": func.__name__},
                    )
                    return result
                except Exception as e:
                    record_metric(
                        "function.execution.errors",
                        1,
                        {
                            "function": func.__name__,
                            "error_type": type(e).__name__,
                        },
                    )
                    raise

        return wrapper

    return decorator


def record_metric(
    name: str,
    value: float,
    attributes: Optional[Dict[str, str]] = None,
    metric_type: str = "histogram",
):
    """
    Record a metric value.

    Args:
        name: Metric name
        value: Metric value
        attributes: Metric attributes/labels
        metric_type: Type of metric ("counter", "gauge", "histogram")
    """
    if not _otel_available or not _is_otel_enabled() or not _meter:
        return

    try:
        attrs = attributes or {}

        if metric_type == "counter":
            counter = _meter.create_counter(name)
            counter.add(value, attributes=attrs)
        elif metric_type == "gauge":
            gauge = _meter.create_up_down_counter(name)
            gauge.add(value, attributes=attrs)
        elif metric_type == "histogram":
            histogram = _meter.create_histogram(name)
            histogram.record(value, attributes=attrs)
    except Exception:
        pass


def get_current_trace_context() -> Optional[Dict[str, str]]:
    """Get current trace context for propagation."""
    if not _otel_available or not _is_otel_enabled():
        return None

    try:
        propagator = TraceContextTextMapPropagator()
        context_dict = {}
        propagator.inject(context_dict)
        return context_dict
    except Exception:
        return None


def set_trace_context(context: Dict[str, str]):
    """Set trace context from external source (for distributed tracing)."""
    if not _otel_available or not _is_otel_enabled():
        return

    try:
        propagator = TraceContextTextMapPropagator()
        propagator.extract(context)
    except Exception:
        pass


def log_event(
    message: str,
    level: str = "INFO",
    attributes: Optional[Dict[str, Any]] = None,
):
    """
    Log an event with OpenTelemetry logging.

    Args:
        message: Log message
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        attributes: Additional attributes
    """
    if not _otel_available or not _is_otel_enabled() or not _logger:
        logger.log(level, message)
        return

    try:
        from opentelemetry._logs import SeverityNumber

        severity_map = {
            "DEBUG": SeverityNumber.DEBUG,
            "INFO": SeverityNumber.INFO,
            "WARNING": SeverityNumber.WARNING,
            "ERROR": SeverityNumber.ERROR,
            "CRITICAL": SeverityNumber.CRITICAL,
        }

        _logger.emit(
            _logs.LogRecord(
                body=message,
                severity_number=severity_map.get(level, SeverityNumber.INFO),
                attributes=attributes or {},
            )
        )
    except Exception:
        logger.log(level, message)

