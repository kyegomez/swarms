"""
Autosave Directory Structure Example

This example demonstrates how the autosave feature organizes files in
agent-specific directories. Each agent gets its own isolated workspace.

Directory structure:
    workspace_dir/
    └── agents/
        └── {agent_name}-{uuid12}/
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
        model_name="gpt-5.4",
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
for agent in agents:
    agent_workspace = agent._get_agent_workspace_dir()
    for file in sorted(os.listdir(agent_workspace)):
        file_path = os.path.join(agent_workspace, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
