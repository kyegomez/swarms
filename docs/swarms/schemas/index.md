# Schemas

The `schemas` package defines typed models used across agent execution, MCP integrations, and API exchange formats.

## Public API

From `swarms.schemas`:

- `Step`
- `ManySteps`
- `MCPConnection`
- `MultipleMCPConnections`

## Core Models

### Step and ManySteps

`Step` represents one agent step with identifiers, timestamp, and structured response payload.

`ManySteps` represents run-level execution history and metadata, including:

- Agent identity
- Task and run identifiers
- Step list
- Token usage and stop details
- Full run history text

### MCPConnection Models

`MCPConnection` captures one MCP endpoint configuration:

- URL, transport, timeout, headers
- Optional auth token
- Optional tool configuration map

`MultipleMCPConnections` groups several MCP connections for multi-server workflows.

## Quickstart

```python
from swarms.schemas import MCPConnection, MultipleMCPConnections

conn = MCPConnection(
    url="http://localhost:8000/mcp",
    transport="streamable_http",
    timeout=10,
)

connections = MultipleMCPConnections(connections=[conn])
print(connections.model_dump())
```

## Notes

- Prefer schema models over untyped dictionaries for agent/MCP pipelines.
- Validate serialized payloads at boundaries (API, queue, storage).
- Keep transport and timeout aligned with server runtime characteristics.
