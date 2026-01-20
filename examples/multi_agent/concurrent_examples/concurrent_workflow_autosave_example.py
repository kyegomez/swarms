#!/usr/bin/env python3
"""
ConcurrentWorkflow Autosave Example

This example demonstrates how to use the autosave feature to automatically
save conversation history after workflow execution.

Usage:
    python concurrent_workflow_autosave_example.py
"""

from swarms import Agent, ConcurrentWorkflow


def main():
    """Example: Using ConcurrentWorkflow with autosave enabled."""
    
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
    
    # Create workflow with autosave enabled
    # No need to set WORKSPACE_DIR - it defaults to 'agent_workspace' automatically!
    workflow = ConcurrentWorkflow(
        name="content-workflow",
        description="Content creation workflow",
        agents=[writer, editor],
        autosave=True,  # Enable autosave - conversation will be saved automatically
    )
    
    print(f"Workspace directory: {workflow.swarm_workspace_dir}")
    
    # Run a task
    result = workflow.run("Write a short paragraph about artificial intelligence.")
    
    # Conversation history is automatically saved to:
    # {workspace_dir}/swarms/ConcurrentWorkflow/content-workflow-{timestamp}/conversation_history.json
    print(f"\n✅ Task completed! Conversation saved to:")
    print(f"   {workflow.swarm_workspace_dir}/conversation_history.json")


if __name__ == "__main__":
    # No setup needed - autosave works out of the box!
    # WORKSPACE_DIR defaults to 'agent_workspace' if not set
    main()
