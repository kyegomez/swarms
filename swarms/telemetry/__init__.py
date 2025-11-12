from swarms.telemetry.main import (
    generate_user_id,
    get_machine_id,
    get_comprehensive_system_info,
    log_agent_data,
)

try:
    from swarms.telemetry.opentelemetry_integration import (
        get_tracer,
        get_meter,
        trace_span,
        trace_function,
        record_metric,
        get_current_trace_context,
        set_trace_context,
        log_event,
    )
    __all__ = [
        "generate_user_id",
        "get_machine_id",
        "get_comprehensive_system_info",
        "log_agent_data",
        "get_tracer",
        "get_meter",
        "trace_span",
        "trace_function",
        "record_metric",
        "get_current_trace_context",
        "set_trace_context",
        "log_event",
    ]
except ImportError:
    __all__ = [
        "generate_user_id",
        "get_machine_id",
        "get_comprehensive_system_info",
        "log_agent_data",
    ]
