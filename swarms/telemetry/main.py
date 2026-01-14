import datetime
import hashlib
import os
import platform
import socket
import threading
import uuid
from typing import Any, Dict

import psutil
import requests
from functools import lru_cache

# Optional OpenTelemetry support (best-effort)
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )

    _OPENTELEMETRY_AVAILABLE = True
except Exception:
    _OPENTELEMETRY_AVAILABLE = False


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


def _sanitize_agent_payload(data_dict: dict) -> Dict[str, Any]:
    """Return a small, non-sensitive representation of an agent for telemetry.

    Avoid shipping secrets (API keys, full llm args, etc.).
    """
    if not isinstance(data_dict, dict):
        return {"note": "invalid-agent-payload"}

    def _safe_list(v):
        if isinstance(v, (list, tuple)):
            return list(v)[:50]
        return v

    # Try common attribute names used in Agent.to_dict()
    agent_name = data_dict.get("agent_name") or data_dict.get("name")
    description = data_dict.get("agent_description") or data_dict.get("description")
    tags = _safe_list(data_dict.get("tags") or data_dict.get("capabilities") or [])
    capabilities = _safe_list(data_dict.get("capabilities") or [])
    tools = data_dict.get("tools") or data_dict.get("tools_list_dictionary") or []
    handoffs = data_dict.get("handoffs")

    # Extract handoff agent names if possible
    handoff_names = []
    try:
        if isinstance(handoffs, dict):
            handoff_names = list(handoffs.keys())
        elif isinstance(handoffs, (list, tuple)):
            for h in handoffs:
                if isinstance(h, dict) and "agent_name" in h:
                    handoff_names.append(h.get("agent_name"))
                else:
                    handoff_names.append(getattr(h, "agent_name", str(h)))
    except Exception:
        handoff_names = []

    sanitized = {
        "id": data_dict.get("id"),
        "agent_name": agent_name,
        "description": (description[:240] if isinstance(description, str) else description),
        "role": data_dict.get("role"),
        "tags": tags,
        "capabilities": capabilities,
        "model_name": data_dict.get("model_name") or data_dict.get("model"),
        "tools_count": len(tools) if tools is not None else 0,
        "handoff_agents": handoff_names,
        "workspace_dir_present": bool(data_dict.get("workspace_dir")),
        "telemetry_version": "v1",
        "reported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    return sanitized


def _log_agent_data(data_dict: dict):
    """Simple function to log agent data using requests library"""

    # Allow telemetry endpoint and API key to be configured via environment
    url = os.getenv(
        "SWARMS_TELEMETRY_URL", "https://swarms.world/api/get-agents/log-agents"
    )
    api_key = os.getenv("SWARMS_TELEMETRY_API_KEY")

    # Build a structured, sanitized telemetry payload for agents
    agent_payload = _sanitize_agent_payload(data_dict)

    log = {
        "agent": agent_payload,
        "system_data": get_comprehensive_system_info(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    payload = {"data": log}

    headers = {"Content-Type": "application/json"}
    if api_key:
        # Allow both raw key or already prefixed Bearer
        headers["Authorization"] = api_key if api_key.startswith("Bearer ") else f"Bearer {api_key}"

    # If OpenTelemetry is available, emit an event/span so observability systems
    # can pick up and index agent creation/updates quickly. This is best-effort.
    try:
        if _OPENTELEMETRY_AVAILABLE:
            tracer = _init_tracer()
            with tracer.start_as_current_span("swarms.agent.telemetry") as span:
                # Add attributes for quick search in tracing backends
                try:
                    span.set_attribute("agent.id", str(data_dict.get("id")))
                    span.set_attribute("agent.name", str(data_dict.get("agent_name") or data_dict.get("name")))
                    span.set_attribute(
                        "agent.type", str(data_dict.get("type", "unknown"))
                    )
                    span.add_event("agent.logged", attributes={"payload": "shallow"})
                except Exception:
                    pass

    except Exception:
        # OpenTelemetry failures should not block normal telemetry
        pass

    # Send HTTP POST in background to avoid blocking agent creation paths
    def _post():
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=8)
            # Silently ignore non-200 responses, don't raise in library code
            return response.status_code
        except Exception:
            return None

    t = threading.Thread(target=_post, daemon=True)
    t.start()


def log_agent_data(data_dict: dict):
    try:
        _log_agent_data(data_dict)
    except Exception:
        pass


def _init_tracer():
    """Initialize and return an OpenTelemetry tracer (cached)."""
    # Use a simple caching approach on the module
    global _TRACER
    try:
        _TRACER  # type: ignore
        return _TRACER
    except Exception:
        pass

    if not _OPENTELEMETRY_AVAILABLE:
        # Fallback to noop tracer
        return trace.get_tracer(__name__)

    try:
        # Configure OTLP exporter if endpoint provided
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        resource = Resource.create({"service.name": "swarms-telemetry"})
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)

        if otlp_endpoint:
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            span_processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(span_processor)

        _TRACER = trace.get_tracer(__name__)
        return _TRACER
    except Exception:
        return trace.get_tracer(__name__)
