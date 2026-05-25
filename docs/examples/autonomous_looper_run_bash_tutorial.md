# Autonomous Looper Run Bash Tutorial

The autonomous looper lets an `Agent` plan a task, execute subtasks, and produce a final summary when `max_loops="auto"`. The `run_bash` tool can be enabled when an agent needs controlled terminal access for local inspection, scripts, or developer workflows.

Use this pattern carefully. Terminal access is powerful, so only enable `run_bash` for trusted workloads and keep the tool list as small as possible.

## Example

```python
from swarms import Agent


agent = Agent(
    agent_name="Terminal-Agent",
    agent_description="Agent that can plan tasks and run terminal commands",
    model_name="anthropic/claude-sonnet-4-5",
    dynamic_temperature_enabled=True,
    dynamic_context_window=True,
    max_loops="auto",
    selected_tools=[
        "create_plan",
        "think",
        "subtask_done",
        "complete_task",
        "respond_to_user",
        "read_file",
        "list_directory",
        "run_bash",
    ],
)

result = agent.run(
    "List the current directory and summarize the project structure."
)

print(result)
```

The repository also includes a runnable source example at `examples/single_agent/full_autonomy/example_autonomous_looper_run_bash.py`.

## Tool Selection

The `selected_tools` list controls which autonomous-loop tools are available. Keep this list explicit:

- `create_plan` lets the agent break a task into subtasks.
- `think` gives the agent a reasoning step before acting.
- `read_file` and `list_directory` support local codebase inspection.
- `run_bash` allows terminal commands.
- `subtask_done`, `complete_task`, and `respond_to_user` let the loop report progress and finish.

## Safety Checklist

- Use the smallest tool list that can complete the task.
- Avoid running autonomous terminal workflows on untrusted repositories.
- Review generated commands before using this pattern for destructive operations.
- Keep secrets out of files and environment variables visible to the task.
- Prefer read-only commands for exploration.
- Run the workflow in a disposable workspace for risky experiments.

## Related Docs

- [Agent Reference](../swarms/structs/agent.md)
- [CLI Examples](../swarms/cli/cli_examples.md)
- [Tool System](../swarms/tools/main.md)
