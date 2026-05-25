# Agent on MCP

Model Context Protocol (MCP) lets an agent workflow expose tools or capabilities to MCP-compatible clients. Use this pattern when you want another application to call a Swarms-backed capability through a standard tool interface.

The Swarms docs include MCP client utilities in the [MCP client reference](../swarms/tools/mcp_client_call.md). This page shows the deployment shape for wrapping an agent workflow as a service.

## When to Use MCP

- You need a reusable agent capability that can be called by multiple clients.
- You want a stable tool boundary around an internal workflow.
- You want to separate the agent runtime from the UI or orchestration layer.
- You need to expose selected actions without exposing the full application.

## Service Pattern

An MCP-style deployment usually has three pieces:

1. A Swarms agent or swarm that performs the work.
2. A small server that exposes approved tools.
3. Client configuration that points an MCP-compatible client at the server.

Keep the exported tools narrow and explicit. Each tool should validate inputs, call the agent with bounded settings, and return a compact result.

## Example Tool Wrapper

```python
from swarms import Agent


agent = Agent(
    agent_name="MCP-Research-Agent",
    model_name="gpt-4.1",
    max_loops=1,
)


def run_research_task(topic: str) -> str:
    """Run a bounded research task for an MCP client."""
    return str(agent.run(f"Research this topic and return concise notes: {topic}"))
```

The wrapper can then be registered with the MCP server framework used by your deployment environment.

## Deployment Checklist

- Expose only the tools the client needs.
- Validate all arguments before calling the agent.
- Keep loop counts and tool calls bounded.
- Store credentials in environment variables.
- Add request logging and error handling around every exported tool.
- Avoid returning secrets, raw stack traces, or oversized intermediate context.
