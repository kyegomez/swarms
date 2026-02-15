# Telemetry

The telemetry module provides optional runtime logging for agent usage and host-level system metadata.

## Opt-In Controls

Telemetry is controlled by environment variables:

- `SWARMS_TELEMETRY_ON`: set to `true` or `True` to enable telemetry.
- `SWARMS_API_KEY`: API key sent as the `Authorization` header.

If telemetry is not enabled, calls to `log_agent_data()` are no-ops.

## Public API

- `log_agent_data(data_dict: dict)`: Guarded entry point that checks telemetry opt-in.
- `_log_agent_data(data_dict: dict)`: Internal sender that posts data to the telemetry endpoint.
- `get_comprehensive_system_info()`: Cached hardware/system metadata collector.

## Payload Shape

Telemetry payload includes:

- Agent data (`data_dict`)
- System metadata (platform, CPU, memory summary, hostname, etc.)
- UTC timestamp

## Quickstart

```python
import os
from swarms.telemetry.main import log_agent_data

os.environ["SWARMS_TELEMETRY_ON"] = "true"
os.environ["SWARMS_API_KEY"] = "<your-api-key>"

log_agent_data(
    {
        "agent_name": "ops-agent",
        "event": "run_complete",
        "latency_ms": 381,
    }
)
```

## Operational Guidance

- Enable telemetry explicitly in environments where collection is approved.
- Avoid including sensitive prompt content unless your policy allows it.
- Rotate API keys regularly and scope usage by environment.
