"""
Autosave Config Access Example

This example demonstrates how to access and use the auto-saved configuration.
The config.json file contains the complete agent configuration and can be
used to recreate the agent or inspect its state.
"""

from swarms import Agent
import os
import json

# Create an agent with autosave enabled
agent = Agent(
    model_name="gpt-4o-mini",
    agent_name="config-access-demo",
    max_loops=2,
    autosave=True,
    verbose=False,
    print_on=False,
)

# Run a task to generate the config file
agent.run("What is 2+2?")

# Get the agent-specific workspace directory
agent_workspace = agent._get_agent_workspace_dir()
config_path = os.path.join(agent_workspace, "config.json")

# Load the saved configuration
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        saved_config = json.load(f)

    # Access autosave metadata
    if "_autosave_metadata" in saved_config:
        metadata = saved_config["_autosave_metadata"]
        timestamp = metadata.get("timestamp")
        loop_count = metadata.get("loop_count")
        agent_id = metadata.get("agent_id")
        agent_name = metadata.get("agent_name")

    # Access agent configuration
    model_name = saved_config.get("model_name")
    max_loops = saved_config.get("max_loops")
    temperature = saved_config.get("temperature")

    # The config can be used to recreate the agent or inspect its state
    # For example, you could create a new agent with the same config:
    # new_agent = Agent(**{k: v for k, v in saved_config.items() if k != '_autosave_metadata'})
