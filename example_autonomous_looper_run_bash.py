"""
Example: Autonomous looper using the run_bash tool

This example shows an agent with max_loops="auto" that can execute
terminal commands via the run_bash tool. The agent plans and runs
shell commands to complete the task.
"""

from swarms import Agent

# Agent with autonomous looping and terminal (bash) access
agent = Agent(
    agent_name="Terminal-Agent",
    agent_description="Agent that can plan tasks and run bash commands on the terminal",
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
        "read_file",
        "list_directory",
        "run_bash",
    ],
    top_p=None,
)

if __name__ == "__main__":
    result = agent.run(
        task="Use the terminal to list the current directory, and see what files are in it."
    )
    print(result)
