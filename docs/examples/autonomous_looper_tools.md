# Autonomous Looper Tools Configuration

The `selected_tools` parameter allows you to configure which tools are available to the agent when using the autonomous looper mode (`max_loops="auto"`).

## Overview

When an agent is set to `max_loops="auto"`, it enters autonomous loop mode where it can:
1. Create a plan by breaking down tasks into subtasks
2. Execute each subtask using available tools
3. Generate a comprehensive summary when complete

The `selected_tools` parameter gives you fine-grained control over which tools the agent can use during this autonomous execution. By default, all tools are enabled (`selected_tools="all"`).

## Available Tools

| Tool Name | Description |
|-----------|-------------|
| `create_plan` | Create a detailed plan for completing a task |
| `think` | Analyze current situation and decide next actions |
| `subtask_done` | Mark a subtask as completed and move to the next task |
| `complete_task` | Mark the main task as complete with comprehensive summary |
| `respond_to_user` | Send messages or responses to the user |
| `create_file` | Create a new file with specified content |
| `update_file` | Update an existing file (replace or append) |
| `read_file` | Read the contents of a file |
| `list_directory` | List files and directories in a path |
| `delete_file` | Delete a file (use with caution) |
| `run_bash` | Execute bash/shell commands on the terminal (returns stdout/stderr) |
| `create_sub_agent` | Create specialized sub-agents for delegation |
| `assign_task` | Assign tasks to sub-agents for asynchronous execution |

## Usage

### Default Behavior (All Tools Available)

```python
from swarms import Agent

agent = Agent(
    agent_name="Full-Access-Agent",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops="auto",
    selected_tools="all",  # Default - all tools available
)
```

### Restricted Tools

```python
agent = Agent(
    agent_name="Planning-Only-Agent",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops="auto",
    selected_tools=[
        "create_plan",
        "think",
        "subtask_done",
        "complete_task",
        "respond_to_user",
    ],
)
```

### File Operations Enabled

```python
agent = Agent(
    agent_name="File-Operations-Agent",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops="auto",
    selected_tools=[
        "create_plan",
        "think",
        "subtask_done",
        "complete_task",
        "respond_to_user",
        "create_file",
        "update_file",
        "read_file",
        "list_directory",
    ],
)
```

### File Operations + Terminal (run_bash)

Enable file operations and terminal command execution:

```python
agent = Agent(
    agent_name="File-and-Terminal-Agent",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops="auto",
    selected_tools=[
        "create_plan",
        "think",
        "subtask_done",
        "complete_task",
        "respond_to_user",
        "create_file",
        "update_file",
        "read_file",
        "list_directory",
        "run_bash",
    ],
)
```

### Minimal Configuration

```python
agent = Agent(
    agent_name="Minimal-Agent",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops="auto",
    selected_tools=[
        "create_plan",
        "subtask_done",
        "complete_task",
    ],
)
```

## Use Cases

### Research Agent (No File Operations)
For agents focused on research and analysis without needing to create or modify files:

```python
research_agent = Agent(
    agent_name="Research-Agent",
    max_loops="auto",
    selected_tools=[
        "create_plan",
        "think",
        "subtask_done",
        "complete_task",
        "respond_to_user",
    ],
)
```

### Code Generation Agent (With File Operations)
For agents that need to create and modify code files:

```python
code_agent = Agent(
    agent_name="Code-Generator",
    max_loops="auto",
    selected_tools=[
        "create_plan",
        "think",
        "subtask_done",
        "complete_task",
        "respond_to_user",
        "create_file",
        "update_file",
        "read_file",
        "list_directory",
    ],
)
```

### Data Analysis Agent (Read-Only Files)
For agents that need to read files but shouldn't modify them:

```python
analysis_agent = Agent(
    agent_name="Data-Analyst",
    max_loops="auto",
    selected_tools=[
        "create_plan",
        "think",
        "subtask_done",
        "complete_task",
        "respond_to_user",
        "read_file",
        "list_directory",
    ],
)
```

### Terminal / DevOps Agent (With run_bash)
For agents that need to run shell commands (e.g. scripts, CLI tools, git):

```python
terminal_agent = Agent(
    agent_name="Terminal-Agent",
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
```

See [Run Bash Tool Tutorial](autonomous_looper_run_bash_tutorial.md) for a step-by-step guide.

## Best Practices

1. **Start Restrictive**: Begin with a minimal set of tools and add more as needed
2. **Security**: Avoid giving file deletion capabilities unless absolutely necessary
3. **Task Alignment**: Choose tools that align with the agent's primary purpose
4. **Testing**: Test your agent with the tool configuration before production use

## Notes

- `selected_tools` defaults to `"all"`, which enables all tools
- When set to a list, the agent will only have access to the tools you specify
- Tool handlers are automatically filtered based on your configuration
- Invalid tool names are ignored (only valid tool names from the list above are used)

## See Also

- [Run Bash Tool Tutorial](autonomous_looper_run_bash_tutorial.md) – Using `run_bash` to execute terminal commands
- [Autonomous Loop Documentation](./autonomous_loop.md)
- [Agent Configuration Guide](./agent_configuration.md)
- [Tool System Overview](./tools.md)
