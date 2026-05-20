import os
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from typing import Any, Dict, Optional

from loguru import logger


def _is_truthy(value: Optional[str]) -> bool:
    return str(value).lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _setup_tracer():
    if not _is_truthy(os.getenv("SWARMS_OTEL_ENABLED")):
        return None

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
        )
    except Exception:
        logger.warning(
            "OpenTelemetry requested but SDK not installed. "
            "Install opentelemetry-sdk and opentelemetry-exporter-otlp."
        )
        return None

    service_name = os.getenv(
        "SWARMS_OTEL_SERVICE_NAME", "swarms-agent"
    )
    endpoint = os.getenv(
        "SWARMS_OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://localhost:4318/v1/traces",
    )
    insecure = _is_truthy(os.getenv("SWARMS_OTEL_INSECURE", "true"))

    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name})
    )
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=insecure)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return trace.get_tracer("swarms.telemetry")


@contextmanager
def start_span(
    name: str, attributes: Optional[Dict[str, Any]] = None
):
    tracer = _setup_tracer()
    if tracer is None:
        with nullcontext():
            yield None
        return

    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, value)
        yield span
