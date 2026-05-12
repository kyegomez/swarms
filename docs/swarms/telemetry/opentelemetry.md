# OpenTelemetry Tracing

Swarms can emit OpenTelemetry traces for agent and multi-agent runs. Tracing is disabled by default so local development and existing deployments keep the same behavior until you opt in.

## Install exporters

The package includes OpenTelemetry dependencies. If you are working from source, install the project dependencies before enabling tracing:

```bash
poetry install
```

## Enable tracing

Set `SWARMS_OTEL_ENABLED=true` to enable spans. To export spans to an OTLP collector, set `SWARMS_OTEL_EXPORTER_OTLP_ENDPOINT` to the collector HTTP endpoint:

```bash
export SWARMS_OTEL_ENABLED=true
export SWARMS_OTEL_SERVICE_NAME=swarms-production
export SWARMS_OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1/traces
```

If `SWARMS_OTEL_EXPORTER_OTLP_ENDPOINT` is present, tracing is enabled automatically. When no endpoint is configured, Swarms uses the active OpenTelemetry provider, which lets applications configure their own exporter before importing or running agents.

## What gets traced

Swarms creates spans around the main `run` entrypoints for:

- `Agent`
- `SwarmRouter`
- `SequentialWorkflow`
- `ConcurrentWorkflow`
- `AgentRearrange`
- `GroupChat`
- `GraphWorkflow`

Each span includes the Swarms component, the configured instance name, the function name, and a success or error status. Exceptions are recorded on the active span before they are raised again.

## Privacy model

Tracing does not attach prompt text, task content, tool arguments, images, responses, or conversation history as span attributes. This keeps observability useful for production debugging while avoiding accidental export of sensitive user data.

## Example

```python
import os

from swarms import Agent

os.environ["SWARMS_OTEL_ENABLED"] = "true"
os.environ["SWARMS_OTEL_EXPORTER_OTLP_ENDPOINT"] = (
    "http://localhost:4318/v1/traces"
)

agent = Agent(
    agent_name="researcher",
    system_prompt="You summarize technical documents.",
    model_name="gpt-4o-mini",
)

agent.run("Summarize the release notes.")
```

Your collector should receive a span named `swarms.agent.run` with the `swarms.component`, `swarms.instance_name`, `swarms.function`, and `swarms.status` attributes.
