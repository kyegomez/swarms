#!/usr/bin/env python3
"""
HierarchicalSwarm Autosave Example

This example demonstrates how to use the autosave feature to automatically
save conversation history after swarm execution.

Usage:
    python hierarchical_swarm_autosave_example.py
"""

import os
from swarms import Agent, HierarchicalSwarm


def main():
    """Example: Using HierarchicalSwarm with autosave enabled."""
    
    # Create worker agents
    writer = Agent(
        agent_name="Writer",
        agent_description="Content writer",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
    )
    
    editor = Agent(
        agent_name="Editor",
        agent_description="Content editor",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
    )
    
    # Create swarm with autosave enabled
    swarm = HierarchicalSwarm(
        name="content-team",
        description="Content creation team",
        agents=[writer, editor],
        max_loops=1,
        autosave=True,  # Enable autosave - conversation will be saved automatically
    )
    
    print(f"Workspace directory: {swarm.swarm_workspace_dir}")
    
    # Run a task
    result = swarm.run("Write a short paragraph about artificial intelligence.")
    
    # Conversation history is automatically saved to:
    # {workspace_dir}/swarms/HierarchicalSwarm/content-team-{timestamp}/conversation_history.json
    print(f"\n✅ Task completed! Conversation saved to:")
    print(f"   {swarm.swarm_workspace_dir}/conversation_history.json")


if __name__ == "__main__":
    # Set WORKSPACE_DIR if not already set
    if not os.getenv("WORKSPACE_DIR"):
        os.environ["WORKSPACE_DIR"] = "agent_workspace"
    
    main()
