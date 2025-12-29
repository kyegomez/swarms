"""
Autosave Recovery Example

This example demonstrates how autosave helps recover from interruptions.
When an agent run is interrupted (KeyboardInterrupt) or encounters an error,
the configuration is automatically saved, allowing you to inspect what happened.

The config.json file is updated at each step, so you can see the agent's state
even if it was interrupted mid-execution.
"""

from swarms import Agent
import os
import json

# Create an agent with autosave enabled
agent = Agent(
    model_name="gpt-4o-mini",
    agent_name="recovery-demo",
    max_loops=2,
    autosave=True,
    verbose=False,
    print_on=False,
)

# Get the agent-specific workspace directory
agent_workspace = agent._get_agent_workspace_dir()

# Simulate an interrupted run
# In a real scenario, you might press Ctrl+C or an error might occur
try:
    # Start a task that might be interrupted
    response = agent.run(
        "Count from 1 to 10, explaining each number."
    )
except KeyboardInterrupt:
    # The autosave will have already saved the config on interrupt
    pass

# Check if config.json exists and load it
config_path = os.path.join(agent_workspace, "config.json")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
        if "_autosave_metadata" in config:
            metadata = config["_autosave_metadata"]

# The agent state can also be loaded
state_files = [
    f
    for f in os.listdir(agent_workspace)
    if "state" in f and f.endswith(".json")
]
