import datetime
import hashlib
import platform
import socket
import uuid
import os
import functools
import inspect
import time
from typing import (
    Any,
    Dict,
    Callable,
    Optional,
)

import psutil
import requests
from functools import lru_cache


# Helper functions
def generate_user_id():
    """Generate user id

    Returns:
        _type_: _description_
    """
    return str(uuid.uuid4())


def get_machine_id():
    """Get machine id

    Returns:
        _type_: _description_
    """
    raw_id = platform.node()
    hashed_id = hashlib.sha256(raw_id.encode()).hexdigest()
    return hashed_id


@lru_cache(maxsize=1)
def get_comprehensive_system_info() -> Dict[str, Any]:
    # Basic platform and hardware information
    system_data = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "platform_full": platform.platform(),
        "architecture": platform.machine(),
        "architecture_details": platform.architecture()[0],
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
    }

    # MAC address
    try:
        system_data["mac_address"] = ":".join(
            [
                f"{(uuid.getnode() >> elements) & 0xFF:02x}"
                for elements in range(0, 2 * 6, 8)
            ][::-1]
        )
    except Exception as e:
        system_data["mac_address"] = f"Error: {str(e)}"

    # CPU information
    system_data["cpu_count_logical"] = psutil.cpu_count(logical=True)
    system_data["cpu_count_physical"] = psutil.cpu_count(
        logical=False
    )

    # Memory information
    vm = psutil.virtual_memory()
    total_ram_gb = vm.total / (1024**3)
    used_ram_gb = vm.used / (1024**3)
    free_ram_gb = vm.free / (1024**3)
    available_ram_gb = vm.available / (1024**3)

    system_data.update(
        {
            "memory_total_gb": f"{total_ram_gb:.2f}",
            "memory_used_gb": f"{used_ram_gb:.2f}",
            "memory_free_gb": f"{free_ram_gb:.2f}",
            "memory_available_gb": f"{available_ram_gb:.2f}",
            "memory_summary": f"Total: {total_ram_gb:.2f} GB, Used: {used_ram_gb:.2f} GB, Free: {free_ram_gb:.2f} GB, Available: {available_ram_gb:.2f} GB",
        }
    )

    # Python version
    system_data["python_version"] = platform.python_version()

    # Generate unique identifier based on system info
    try:
        unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(system_data))
        system_data["unique_identifier"] = str(unique_id)
    except Exception as e:
        system_data["unique_identifier"] = f"Error: {str(e)}"

    return system_data


def _log_agent_data(data_dict: dict):
    """
    Logs agent data and system information to the swarms.world telemetry endpoint via a POST request.

    This function is a low-level, internal utility that sends the provided agent state along with current
    system telemetry to the Swarms service for analytics and diagnostics. Data includes a timestamp,
    comprehensive system information, and the state of the agent as passed in `data_dict`.

    Args:
        data_dict (dict): Dictionary representing the current agent's state/config/data.

    Side Effects:
        Sends a POST request to the Swarms telemetry endpoint.
        Does not raise exceptions on failed request (silent fail).

    Security Warning:
        The authorization key is included in the request header.
        Remove or rotate keys as necessary for production security.

    Returns:
        None
    """
    url = "https://swarms.world/api/get-agents/log-agents"

    log = {
        "data": data_dict,
        "system_data": get_comprehensive_system_info(),
        "timestamp": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
    }

    payload = {
        "data": log,
    }

    key = os.getenv("SWARMS_API_KEY")

    headers = {
        "Content-Type": "application/json",
        "Authorization": key,
    }

    try:
        response = requests.post(
            url, json=payload, headers=headers, timeout=10
        )

        if response.status_code == 200:
            return
    except Exception:
        pass


def log_agent_data(data_dict: dict):
    """
    Public wrapper to log agent data and telemetry if telemetry is enabled.

    This function checks the 'SWARMS_TELEMETRY' environment variable. If set to the string "true",
    it records agent telemetry using the internal _log_agent_data function.
    Otherwise, it does nothing.

    Args:
        data_dict (dict): Agent data to be transmitted if telemetry is enabled.

    Returns:
        None
    """
    get_telemetry = os.getenv("SWARMS_TELEMETRY_ON")

    if get_telemetry == "True" or get_telemetry == "true":
        _log_agent_data(data_dict)
    else:
        pass


# --- OpenTelemetry (OTel) Integration ---
# Optimized for high-performance and documentation build environment safety.

def setup_telemetry(service_name: str = "swarms"):
    """
    Initializes OpenTelemetry tracing and metrics if an endpoint is configured.

    Args:
        service_name (str): The name of the service for telemetry. Defaults to "swarms".

    Returns:
        None
    """
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not otlp_endpoint:
        return

    try:
        from opentelemetry import (
            trace,
            metrics,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.sdk.resources import (
            Resource,
        )
        from opentelemetry.sdk.trace import (
            TracerProvider,
        )
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
        )
        from opentelemetry.sdk.metrics import (
            MeterProvider,
        )
        from opentelemetry.sdk.metrics.export import (
            PeriodicExportingMetricReader,
        )

        resource = Resource.create({"service.name": service_name})

        # Tracing initialization
        if not isinstance(
            trace.get_tracer_provider(), TracerProvider
        ):
            provider = TracerProvider(resource=resource)
            processor = BatchSpanProcessor(
                OTLPSpanExporter(endpoint=otlp_endpoint)
            )
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)

        # Metrics initialization
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=otlp_endpoint)
        )
        meter_provider = MeterProvider(
            resource=resource, metric_readers=[metric_reader]
        )
        metrics.set_meter_provider(meter_provider)

    except Exception:
        # Fails silently to ensure high availability of the core framework
        pass


def trace_span(name: Optional[str] = None):
    """
    Decorator to wrap a function in an OpenTelemetry span and record metrics.
    Zero-overhead and safe for environments without OpenTelemetry.

    Args:
        name (Optional[str]): The name of the span. Defaults to the function name.

    Returns:
        Callable: The decorated function.
    """

    def decorator(func: Callable):
        span_name = name or func.__name__
        is_async = inspect.iscoroutinefunction(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                from opentelemetry import (
                    trace,
                    metrics,
                )

                tracer = trace.get_tracer("swarms")
                meter = metrics.get_meter("swarms")
            except ImportError:
                return func(*args, **kwargs)

            # Extract metadata attributes
            attributes = {}
            if args and hasattr(args[0], "__class__"):
                instance = args[0]
                attributes["swarms.class"] = (
                    instance.__class__.__name__
                )
                if hasattr(instance, "agent_name"):
                    attributes["swarms.agent_name"] = str(
                        instance.agent_name
                    )
                if hasattr(instance, "model_name"):
                    attributes["swarms.model_name"] = str(
                        instance.model_name
                    )

            start_time = time.time()
            try:
                counter = meter.create_counter("swarms.task.count")
                counter.add(1, attributes)
            except Exception:
                pass

            with tracer.start_as_current_span(
                span_name, attributes=attributes
            ) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_status(trace.StatusCode.OK)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(
                        trace.StatusCode.ERROR, str(e)
                    )
                    raise
                finally:
                    try:
                        duration = (time.time() - start_time) * 1000
                        hist = meter.create_histogram(
                            "swarms.task.duration"
                        )
                        hist.record(duration, attributes)
                    except Exception:
                        pass

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                from opentelemetry import (
                    trace,
                    metrics,
                )

                tracer = trace.get_tracer("swarms")
                meter = metrics.get_meter("swarms")
            except ImportError:
                return await func(*args, **kwargs)

            attributes = {}
            if args and hasattr(args[0], "__class__"):
                instance = args[0]
                attributes["swarms.class"] = (
                    instance.__class__.__name__
                )
                if hasattr(instance, "agent_name"):
                    attributes["swarms.agent_name"] = str(
                        instance.agent_name
                    )
                if hasattr(instance, "model_name"):
                    attributes["swarms.model_name"] = str(
                        instance.model_name
                    )

            start_time = time.time()
            try:
                counter = meter.create_counter("swarms.task.count")
                counter.add(1, attributes)
            except Exception:
                pass

            with tracer.start_as_current_span(
                span_name, attributes=attributes
            ) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(trace.StatusCode.OK)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(
                        trace.StatusCode.ERROR, str(e)
                    )
                    raise
                finally:
                    try:
                        duration = (time.time() - start_time) * 1000
                        hist = meter.create_histogram(
                            "swarms.task.duration"
                        )
                        hist.record(duration, attributes)
                    except Exception:
                        pass

        return async_wrapper if is_async else wrapper

    return decorator
