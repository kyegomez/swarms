# OpenTelemetry Integration

Swarms provides built-in OpenTelemetry support for distributed tracing and metrics collection across agent and multi-agent workflow executions.

## Installation

OpenTelemetry support is optional. Install the required dependencies:

```bash
# Using pip with extras
pip install swarms[otel]

# Or install packages directly
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc
```

## Configuration

Enable OpenTelemetry tracing via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `SWARMS_OTEL_ENABLED` | Enable/disable tracing (`true`/`false`) | `false` |
| `OTEL_SERVICE_NAME` | Service name for traces | `swarms` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint | None |
| `OTEL_EXPORTER_OTLP_HEADERS` | Headers for OTLP exporter | None |

## Quick Start

```python
import os

# Enable tracing
os.environ["SWARMS_OTEL_ENABLED"] = "true"
os.environ["OTEL_SERVICE_NAME"] = "my-agent-app"
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4317"

from swarms import Agent

agent = Agent(
    agent_name="research-agent",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Traces are automatically created for agent.run()
result = agent.run("What is the capital of France?")
```

## What Gets Traced

### Agent Runs

Each `agent.run()` call creates a span with:

- `agent.name` - Agent's name
- `agent.id` - Agent's unique identifier
- `agent.model` - Model being used
- `agent.max_loops` - Maximum loop configuration
- `run.has_image` - Whether image input was provided
- `run.duration_ms` - Execution time in milliseconds
- `run.status` - `success` or `error`

### Multi-Agent Workflows

SwarmRouter, SequentialWorkflow, and ConcurrentWorkflow executions create spans with:

- `swarm.name` - Workflow name
- `swarm.id` - Workflow identifier
- `swarm.type` - Type of workflow (e.g., `SequentialWorkflow`)
- `swarm.agent_count` - Number of agents in the workflow
- `run.duration_ms` - Total execution time
- `run.status` - `success` or `error`

## Metrics

When enabled, the following metrics are collected:

| Metric | Type | Description |
|--------|------|-------------|
| `swarms.agent.runs` | Counter | Number of agent run invocations |
| `swarms.agent.duration` | Histogram | Duration of agent runs (ms) |
| `swarms.agent.errors` | Counter | Number of agent run errors |
| `swarms.swarm.runs` | Counter | Number of swarm run invocations |
| `swarms.swarm.duration` | Histogram | Duration of swarm runs (ms) |

## Using with Jaeger

```bash
# Start Jaeger
docker run -d --name jaeger \
    -p 16686:16686 \
    -p 4317:4317 \
    jaegertracing/all-in-one:latest

# Configure environment
export SWARMS_OTEL_ENABLED=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Run your application
python your_app.py

# View traces at http://localhost:16686
```

## Custom Tracing

Use the `trace_context` helper for custom spans:

```python
from swarms.telemetry import trace_context

with trace_context("custom.operation", {"key": "value"}) as span:
    # Your code here
    result = do_something()
    
    if span:
        span.set_attribute("result.count", len(result))
```

## Checking Status

```python
from swarms.telemetry import is_otel_enabled, otel_available

# Check if OTEL packages are installed
print(f"OTEL available: {otel_available()}")

# Check if OTEL is enabled
print(f"OTEL enabled: {is_otel_enabled()}")
```

## Best Practices

1. **Production environments**: Always set `OTEL_EXPORTER_OTLP_ENDPOINT` to send traces to your collector
2. **Sampling**: For high-volume applications, configure sampling in your OTLP collector
3. **Service naming**: Use descriptive `OTEL_SERVICE_NAME` values to identify your application
4. **Error tracking**: Errors are automatically recorded with exception details

## Graceful Degradation

The integration is designed to be non-invasive:

- If OTEL packages are not installed, tracing is silently disabled
- If `SWARMS_OTEL_ENABLED` is not set to `true`, no tracing overhead is added
- All tracing operations are wrapped in try/except to prevent affecting normal execution
