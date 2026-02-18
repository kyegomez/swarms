# Run Bash Tool Tutorial

Use the **`run_bash`** tool so an autonomous agent can run shell commands on the terminal.

## When to use it

- The agent needs to run CLI commands (e.g. `ls`, `python script.py`, `git status`).
- You use `max_loops="auto"` and want the agent to have terminal access.
- You want to keep other tools restricted and only add bash execution.

## Enable the tool

Include `"run_bash"` in `selected_tools` when creating the agent:

```python
from swarms import Agent

agent = Agent(
    agent_name="Terminal-Agent",
    agent_description="Agent that can run bash commands on the terminal",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops="auto",
    dynamic_context_window=True,
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
```

## Run a task

The agent will plan and call `run_bash` when it needs to run a command:

```python
result = agent.run(
    task="Use the terminal to list the current directory, then run 'echo Hello from bash' and report the output."
)
print(result)
```

## Tool parameters

| Parameter           | Type    | Description |
|--------------------|---------|-------------|
| `command`           | string  | The bash/shell command to run (e.g. `ls -la`, `python script.py`). |
| `timeout_seconds`  | integer | (Optional) Max seconds to wait; default is 60. |

Commands run in the agent’s workspace directory when available. Stdout and stderr are returned; long-running commands should use a higher `timeout_seconds` or be avoided.

## Example file

A full runnable example is in the repo:

```
examples/single_agent/full_autonomy/example_autonomous_looper_run_bash.py
```

## See also

- [Autonomous Looper Tools](autonomous_looper_tools.md) – Configuring `selected_tools`
- [Agent Reference](../swarms/structs/agent.md) – `selected_tools` and autonomous loop
