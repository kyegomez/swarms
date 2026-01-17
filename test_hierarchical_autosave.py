#!/usr/bin/env python3
"""
Test script to verify HierarchicalSwarm autosave with default workspace directory.

This test verifies that:
1. Autosave works without setting WORKSPACE_DIR (uses default)
2. Conversation history is saved correctly
3. Workspace directory is created automatically
"""

import os
import json
from pathlib import Path
from swarms import Agent, HierarchicalSwarm


def test_autosave_with_default_workspace():
    """Test that autosave works with default workspace directory."""
    
    # Remove WORKSPACE_DIR if it's set to test the default
    original_workspace = os.environ.pop("WORKSPACE_DIR", None)
    
    try:
        print("=" * 70)
        print("Testing HierarchicalSwarm Autosave with Default Workspace")
        print("=" * 70)
        print("\n📝 WORKSPACE_DIR is not set - should use default 'agent_workspace'\n")
        
        # Create agents
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
            name="test-swarm",
            description="Test swarm for autosave",
            agents=[writer, editor],
            max_loops=1,
            autosave=True,
            verbose=True,  # Enable verbose to see default workspace message
        )
        
        # Verify workspace directory was created
        assert swarm.swarm_workspace_dir is not None, "Workspace directory should be created"
        print(f"✅ Workspace directory created: {swarm.swarm_workspace_dir}")
        
        # Verify it's in the expected location (agent_workspace)
        assert "agent_workspace" in swarm.swarm_workspace_dir, "Should use default 'agent_workspace' directory"
        print(f"✅ Using default workspace directory (agent_workspace)")
        
        # Run a task
        print("\n📝 Running task...")
        result = swarm.run("Write a short sentence about Python programming.")
        print("✅ Task completed\n")
        
        # Verify conversation history file exists
        conversation_file = Path(swarm.swarm_workspace_dir) / "conversation_history.json"
        assert conversation_file.exists(), f"Conversation history file should exist at {conversation_file}"
        print(f"✅ Conversation history file created: {conversation_file}")
        
        # Verify file contains valid JSON
        with open(conversation_file, "r") as f:
            conversation = json.load(f)
        
        assert isinstance(conversation, list), "Conversation should be a list"
        assert len(conversation) > 0, "Conversation should contain messages"
        print(f"✅ Conversation history contains {len(conversation)} messages")
        
        print("\n" + "=" * 70)
        print("✅ All tests passed!")
        print("=" * 70)
        print(f"\n📁 Workspace: {swarm.swarm_workspace_dir}")
        print(f"📄 File: {conversation_file}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Restore original WORKSPACE_DIR if it existed
        if original_workspace:
            os.environ["WORKSPACE_DIR"] = original_workspace


if __name__ == "__main__":
    test_autosave_with_default_workspace()
