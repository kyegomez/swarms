"""
Basic Autosave Example

This example demonstrates the basic autosave functionality of the Agent class.
When autosave=True, the agent automatically saves its configuration to a JSON file
at each step of execution, and saves the full state on errors or interruptions.

The files are saved in: workspace_dir/agent-{agent_name}-{uuid}/
"""

from swarms import Agent
import os
import json

# Create an agent with autosave enabled
agent = Agent(
    model_name="gpt-4o-mini",
    agent_name="autosave-demo",
    max_loops=3,
    autosave=True,  # Enable autosave
    verbose=False,
    print_on=False,
)

# Get the agent-specific workspace directory
agent_workspace = agent._get_agent_workspace_dir()

# Run a simple task
response = agent.run(
    "Write a short poem about artificial intelligence."
)

# Check if config.json was created
config_path = os.path.join(agent_workspace, "config.json")
if os.path.exists(config_path):
    # Load and check metadata
    with open(config_path, "r") as f:
        config = json.load(f)
        if "_autosave_metadata" in config:
            metadata = config["_autosave_metadata"]

# Check for state files
state_files = [
    f for f in os.listdir(agent_workspace) if f.endswith(".json")
]
