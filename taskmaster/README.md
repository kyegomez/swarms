# Taskmaster

Taskmaster is a lightweight orchestration repo that delegates tasks to specialized agents, integrates MCP servers, and exposes a CLI for human approvals. It uses the `swarms` package for agent creation and execution.

Goals:

- Keep tasks, generate research, propose actions, and spawn sub-agents for subtasks.
- Integrate MCP servers for tool access.
- CLI-first workflow (via `typer`) for human approvals and control.

Quickstart

```
python -m pip install -e .
taskmaster --help
```
