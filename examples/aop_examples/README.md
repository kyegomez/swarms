# AOP Examples

This directory contains runnable examples that demonstrate AOP (Agents over Protocol) patterns in Swarms: spinning up a simple MCP server, discovering available agents/tools, and invoking agent tools from client scripts.

## What’s inside

- **Top-level demos**
  - [`example_new_agent_tools.py`](./example_new_agent_tools.py): End‑to‑end demo of agent discovery utilities (list/search agents, get details for one or many). Targets an MCP server at `http://localhost:5932/mcp`.
  - [`list_agents_and_call_them.py`](./list_agents_and_call_them.py): Utility helpers to fetch tools from an MCP server and call an agent‑style tool with a task prompt. Defaults to `http://localhost:8000/mcp`.
  - [`get_all_agents.py`](./get_all_agents.py): Minimal snippet to print all tools exposed by an MCP server as JSON. Defaults to `http://0.0.0.0:8000/mcp`.

- **Server**
  - [`server/server.py`](./server/server.py): Simple MCP server entrypoint you can run locally to expose tools/agents for the client examples.

- **Client**
  - [`client/aop_cluster_example.py`](./client/aop_cluster_example.py): Connect to an AOP cluster and interact with agents.
  - [`client/aop_queue_example.py`](./client/aop_queue_example.py): Example of queue‑style task submission to agents.
  - [`client/aop_raw_task_example.py`](./client/aop_raw_task_example.py): Shows how to send a raw task payload without additional wrappers.
  - [`client/aop_raw_client_code.py`](./client/aop_raw_client_code.py): Minimal, low‑level client calls against the MCP endpoint.

- **Discovery**
  - [`discovery/example_agent_communication.py`](./discovery/example_agent_communication.py): Illustrates simple agent‑to‑agent or agent‑to‑service communication patterns.
  - [`discovery/example_aop_discovery.py`](./discovery/example_aop_discovery.py): Demonstrates discovering available agents/tools via AOP.
  - [`discovery/simple_discovery_example.py`](./discovery/simple_discovery_example.py): A pared‑down discovery walkthrough.
  - [`discovery/test_aop_discovery.py`](./discovery/test_aop_discovery.py): Test‑style script validating discovery functionality.

## Prerequisites

- Python environment with project dependencies installed.
- An MCP server running locally (you can use the provided server example).

## Quick start

1. Start a local MCP server (in a separate terminal):

```bash
python examples/aop_examples/server/server.py
```

1. Try discovery utilities (adjust the URL if your server uses a different port):

```bash
# List exposed tools (defaults to http://0.0.0.0:8000/mcp)
python examples/aop_examples/get_all_agents.py

# Fetch tools and call the first agent-like tool (defaults to http://localhost:8000/mcp)
python examples/aop_examples/list_agents_and_call_them.py

# Rich demo of agent info utilities (expects http://localhost:5932/mcp by default)
python examples/aop_examples/example_new_agent_tools.py
```

1. Explore client variants:

```bash
python examples/aop_examples/client/aop_cluster_example.py
python examples/aop_examples/client/aop_queue_example.py
python examples/aop_examples/client/aop_raw_task_example.py
python examples/aop_examples/client/aop_raw_client_code.py
```

## Tips

- **Server URL/port**: Several examples assume `http://localhost:8000/mcp` or `http://localhost:5932/mcp`. If your server runs elsewhere, update the `server_path`/URL variables at the top of the scripts.
- **Troubleshooting**: If a script reports “No tools available”, ensure the MCP server is running and that the endpoint path (`/mcp`) and port match the script.
- **Next steps**: Use these scripts as templates—swap in your own tools/agents, change the search queries, or extend the client calls to fit your workflow.
