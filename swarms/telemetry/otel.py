import os
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from typing import Any, Callable, Dict, Optional


TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
MAX_ATTRIBUTE_LENGTH = 2048

_otel_configured = False


@dataclass
class OpenTelemetrySpanHandle:
    """Small wrapper around an optional OpenTelemetry span."""

    context_manager: Any
    span: Any
    ended: bool = False


class OpenTelemetryTraceContext:
    """Context manager for optional OpenTelemetry spans."""

    def __init__(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.attributes = attributes
        self.handle: Optional[OpenTelemetrySpanHandle] = None

    def __enter__(self) -> Optional[OpenTelemetrySpanHandle]:
        self.handle = start_otel_span(self.name, self.attributes)
        return self.handle

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_value: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        end_otel_span(self.handle, exc_value)
        return False


def _env_is_true(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in TRUE_VALUES


def is_opentelemetry_enabled() -> bool:
    """Return whether Swarms OpenTelemetry tracing is enabled."""
    if _env_is_true("OTEL_SDK_DISABLED"):
        return False
    return _env_is_true("SWARMS_OTEL_ENABLED") or _env_is_true(
        "SWARMS_TELEMETRY_OTEL"
    )


def _safe_attribute_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        if isinstance(value, str):
            return value[:MAX_ATTRIBUTE_LENGTH]
        return value
    return str(value)[:MAX_ATTRIBUTE_LENGTH]


def _clean_attributes(
    attributes: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not attributes:
        return {}

    clean: Dict[str, Any] = {}
    for key, value in attributes.items():
        safe_value = _safe_attribute_value(value)
        if safe_value is not None:
            clean[key] = safe_value
    return clean


def _configure_tracer_provider() -> None:
    """Configure an SDK tracer provider when the optional SDK exists."""
    global _otel_configured
    if _otel_configured:
        return
    _otel_configured = True

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
    except Exception:
        return

    service_name = os.getenv("OTEL_SERVICE_NAME", "swarms")
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    exporter_name = (
        os.getenv("SWARMS_OTEL_EXPORTER")
        or os.getenv("OTEL_TRACES_EXPORTER")
        or (
            "otlp"
            if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            else "console"
        )
    )
    exporter_name = exporter_name.strip().lower()

    exporter = None
    if exporter_name in {"", "none"}:
        exporter = None
    elif "otlp" in exporter_name:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: E501
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter()
        except Exception:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # noqa: E501
                    OTLPSpanExporter,
                )

                exporter = OTLPSpanExporter()
            except Exception:
                exporter = None
    else:
        exporter = ConsoleSpanExporter()

    if exporter is not None:
        provider.add_span_processor(BatchSpanProcessor(exporter))

    try:
        trace.set_tracer_provider(provider)
    except Exception:
        pass


def set_otel_attributes(
    handle_or_span: Optional[Any],
    attributes: Optional[Dict[str, Any]],
) -> None:
    """Set attributes on a span handle without making OTel required."""
    if handle_or_span is None:
        return

    span = getattr(handle_or_span, "span", handle_or_span)
    if span is None:
        return

    for key, value in _clean_attributes(attributes).items():
        try:
            span.set_attribute(key, value)
        except Exception:
            pass


def start_otel_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> Optional[OpenTelemetrySpanHandle]:
    """Start an OpenTelemetry span if env-enabled and installed."""
    if not is_opentelemetry_enabled():
        return None

    try:
        from opentelemetry import trace
    except Exception:
        return None

    try:
        _configure_tracer_provider()
        tracer = trace.get_tracer("swarms")
        context_manager = tracer.start_as_current_span(name)
        span = context_manager.__enter__()
        handle = OpenTelemetrySpanHandle(
            context_manager=context_manager,
            span=span,
        )
        set_otel_attributes(handle, attributes)
        return handle
    except Exception:
        return None


def record_otel_exception(
    handle_or_span: Optional[Any],
    error: BaseException,
) -> None:
    if handle_or_span is None:
        return

    span = getattr(handle_or_span, "span", handle_or_span)
    if span is None:
        return

    try:
        span.record_exception(error)
    except Exception:
        pass

    try:
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.ERROR, str(error)))
    except Exception:
        pass


def end_otel_span(
    handle: Optional[OpenTelemetrySpanHandle],
    error: Optional[BaseException] = None,
) -> None:
    if handle is None or handle.ended:
        return

    handle.ended = True
    if error is not None:
        record_otel_exception(handle, error)

    try:
        exc_type = type(error) if error is not None else None
        exc_tb = error.__traceback__ if error is not None else None
        handle.context_manager.__exit__(exc_type, error, exc_tb)
    except Exception:
        pass


def otel_trace(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> OpenTelemetryTraceContext:
    """Return a context manager for an optional OpenTelemetry span."""
    return OpenTelemetryTraceContext(name=name, attributes=attributes)


def task_attributes(
    task: Any = None,
    img: Optional[str] = None,
    imgs: Optional[Any] = None,
) -> Dict[str, Any]:
    attributes: Dict[str, Any] = {
        "swarms.task.has_image": img is not None or imgs is not None,
        "swarms.task.image_count": len(imgs)
        if isinstance(imgs, (list, tuple))
        else int(img is not None),
    }

    if task is not None:
        task_text = str(task)
        attributes["swarms.task.length"] = len(task_text)
        if _env_is_true("SWARMS_OTEL_RECORD_CONTENT"):
            attributes["swarms.task.content"] = task_text

    return attributes


def agent_span_attributes(
    agent: Any,
    task: Any = None,
    img: Optional[str] = None,
    imgs: Optional[Any] = None,
    n: int = 1,
) -> Dict[str, Any]:
    attributes = {
        "swarms.agent.id": getattr(agent, "id", None),
        "swarms.agent.name": getattr(agent, "agent_name", None)
        or getattr(agent, "name", None),
        "swarms.agent.model": getattr(agent, "model_name", None),
        "swarms.agent.max_loops": getattr(agent, "max_loops", None),
        "swarms.agent.output_count": n,
    }
    attributes.update(task_attributes(task=task, img=img, imgs=imgs))
    return attributes


def method_span_attributes(
    instance: Any,
    method: Callable[..., Any],
    args: tuple,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Build span attributes for swarm/workflow methods."""
    try:
        bound = signature(method).bind_partial(
            instance, *args, **kwargs
        )
        bound_args = dict(bound.arguments)
    except Exception:
        bound_args = dict(kwargs)
        if args:
            bound_args["task"] = args[0]

    task = bound_args.get("task")
    tasks = bound_args.get("tasks")
    img = bound_args.get("img")
    imgs = bound_args.get("imgs")

    attributes = swarm_span_attributes(
        instance,
        task=task,
        tasks=tasks,
        img=img,
        imgs=imgs,
    )
    attributes["swarms.method.name"] = getattr(
        method, "__name__", None
    )
    return attributes


def trace_otel_method(name: str) -> Callable[[Callable], Callable]:
    """Decorate a swarm/workflow run method with optional tracing."""

    def decorator(method: Callable) -> Callable:
        @wraps(method)
        def wrapper(instance: Any, *args: Any, **kwargs: Any) -> Any:
            attributes = method_span_attributes(
                instance,
                method,
                args,
                kwargs,
            )
            with otel_trace(name, attributes) as span:
                output = method(instance, *args, **kwargs)
                set_otel_attributes(
                    span,
                    {"swarms.output.type": type(output).__name__},
                )
                return output

        return wrapper

    return decorator


def swarm_span_attributes(
    swarm: Any,
    task: Any = None,
    tasks: Optional[Any] = None,
    img: Optional[str] = None,
    imgs: Optional[Any] = None,
) -> Dict[str, Any]:
    agents = getattr(swarm, "agents", None) or []
    attributes = {
        "swarms.swarm.id": getattr(swarm, "id", None),
        "swarms.swarm.name": getattr(swarm, "name", None),
        "swarms.swarm.type": swarm.__class__.__name__,
        "swarms.swarm.agent_count": len(agents),
        "swarms.swarm.max_loops": getattr(swarm, "max_loops", None),
        "swarms.swarm.output_type": getattr(swarm, "output_type", None),
    }

    if tasks is not None:
        attributes["swarms.task.batch_size"] = len(tasks)
    attributes.update(task_attributes(task=task, img=img, imgs=imgs))
    return attributes
