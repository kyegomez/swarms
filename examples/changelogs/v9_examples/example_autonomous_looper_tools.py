"""
Example demonstrating the selected_tools parameter

This example shows how to configure which tools are available to the
autonomous looper when max_loops="auto".

Available tools:
- create_plan: Create a detailed plan for completing a task
- think: Analyze current situation and decide next actions
- subtask_done: Mark a subtask as completed
- complete_task: Mark the main task as complete
- respond_to_user: Send messages to the user
- create_file: Create new files
- update_file: Update existing files
- read_file: Read file contents
- list_directory: List directory contents
- delete_file: Delete files
- run_bash: Execute bash/shell commands on the terminal
"""

from swarms import Agent

# Example 1: Agent with all autonomous looper tools (default behavior)
agent_full = Agent(
    agent_name="Full-Access-Agent",
    agent_description="Agent with access to all autonomous looper tools",
    model_name="anthropic/claude-sonnet-4-5",
    dynamic_temperature_enabled=True,
    max_loops="auto",
    dynamic_context_window=True,
    selected_tools="all",  # Default - all tools available
)

# Example 2: Agent with restricted tools (planning and thinking only)
agent_planning_only = Agent(
    agent_name="Planning-Agent",
    agent_description="Agent that can only plan and think, no file operations",
    model_name="anthropic/claude-sonnet-4-5",
    dynamic_temperature_enabled=True,
    max_loops="auto",
    dynamic_context_window=True,
    selected_tools=[
        "create_plan",
        "think",
        "subtask_done",
        "complete_task",
        "respond_to_user",
    ],
)

# Example 3: Agent with file operations and terminal (bash) access
agent_file_ops = Agent(
    agent_name="File-Operations-Agent",
    agent_description="Agent with file operations and terminal execution capabilities",
    model_name="anthropic/claude-sonnet-4-5",
    dynamic_temperature_enabled=True,
    max_loops="auto",
    dynamic_context_window=True,
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

# Example 4: Minimal agent (only core planning tools)
agent_minimal = Agent(
    agent_name="Minimal-Agent",
    agent_description="Agent with minimal autonomous tools",
    model_name="anthropic/claude-sonnet-4-5",
    dynamic_temperature_enabled=True,
    max_loops="auto",
    dynamic_context_window=True,
    selected_tools=[
        "create_plan",
        "subtask_done",
        "complete_task",
    ],
)

# Run examples
if __name__ == "__main__":

    result = agent_planning_only.run(
        task="Research the top 5 programming languages in 2024 and provide a summary of their key features."
    )

    print(result)
