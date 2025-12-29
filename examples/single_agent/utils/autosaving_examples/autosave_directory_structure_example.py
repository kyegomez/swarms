"""
Autosave Directory Structure Example

This example demonstrates how the autosave feature organizes files in
agent-specific directories. Each agent gets its own isolated workspace.

Directory structure:
    workspace_dir/
    └── agent-{agent_name}-{uuid}/
        ├── config.json          # Auto-saved config at each step
        ├── {agent_name}_state.json  # Full agent state
        └── other files...
"""

from swarms import Agent
import os

# Create multiple agents to show directory isolation
agents = []

# Create 3 different agents
for i in range(3):
    agent = Agent(
        model_name="gpt-4o-mini",
        agent_name=f"demo-agent-{i+1}",
        max_loops=2,
        autosave=True,
        verbose=False,
        print_on=False,
    )
    agents.append(agent)

    # Get each agent's workspace directory
    agent_workspace = agent._get_agent_workspace_dir()

    # Run a quick task to generate files
    agent.run(f"Count to {i+1}")

# Check the directory structure
workspace_dir = "agent_workspace"
if os.path.exists(workspace_dir):
    for item in sorted(os.listdir(workspace_dir)):
        item_path = os.path.join(workspace_dir, item)
        if os.path.isdir(item_path):
            # List files in each agent directory
            files = os.listdir(item_path)
            for file in sorted(files):
                file_path = os.path.join(item_path, file)
                size = os.path.getsize(file_path)
