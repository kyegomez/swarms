"""
Simple test script for HierarchicalSwarm autosave feature.

This script demonstrates:
1. Creating a HierarchicalSwarm with autosave enabled
2. Running a task
3. Verifying autosave files are created
4. Displaying the saved files
"""

import json
import os
from pathlib import Path

from swarms import Agent, HierarchicalSwarm
from swarms.utils.workspace_utils import get_workspace_dir


def main():
    """Test HierarchicalSwarm autosave functionality."""
    
    print("=" * 70)
    print("HierarchicalSwarm Autosave Test")
    print("=" * 70)
    
    # Step 1: Create agents
    print("\n📝 Step 1: Creating agents...")
    research_agent = Agent(
        agent_name="Research-Agent",
        agent_description="Research specialist",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )
    
    analysis_agent = Agent(
        agent_name="Analysis-Agent",
        agent_description="Analysis expert",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )
    
    print(f"  ✅ Created 2 agents: {research_agent.agent_name}, {analysis_agent.agent_name}")
    
    # Step 2: Create HierarchicalSwarm with autosave enabled
    print("\n📝 Step 2: Creating HierarchicalSwarm with autosave...")
    swarm = HierarchicalSwarm(
        name="autosave-test-swarm",
        description="Test swarm for autosave feature",
        agents=[research_agent, analysis_agent],
        max_loops=1,
        autosave=True,  # Enable autosave
        autosave_use_timestamp=True,  # Use timestamp in directory name
        verbose=True,  # Enable verbose to see autosave messages
    )
    
    print(f"  ✅ HierarchicalSwarm created")
    print(f"  📁 Workspace directory: {swarm.swarm_workspace_dir}")
    
    # Step 3: Run a simple task
    print("\n📝 Step 3: Running a task...")
    task = "Explain what Python programming is in 2-3 sentences."
    print(f"  Task: {task}")
    
    try:
        result = swarm.run(task)
        print(f"  ✅ Task completed successfully")
        print(f"  📊 Result type: {type(result).__name__}")
    except Exception as e:
        print(f"  ⚠️  Task execution error: {e}")
        result = None
    
    # Step 4: Verify autosave files
    print("\n" + "=" * 70)
    print("Autosave Files Verification")
    print("=" * 70)
    
    if not swarm.swarm_workspace_dir:
        print("  ❌ No workspace directory found. Autosave may not be enabled.")
        return
    
    workspace_path = Path(swarm.swarm_workspace_dir)
    print(f"\n  📁 Workspace: {workspace_path}")
    
    # Expected files
    expected_files = {
        "config.json": "Initial configuration",
        "state.json": "Swarm state after execution",
        "metadata.json": "Execution metadata",
        "conversation_history.json": "Conversation history",
    }
    
    print("\n  📋 Checking files:")
    all_exist = True
    for filename, description in expected_files.items():
        file_path = workspace_path / filename
        exists = file_path.exists()
        status = "✅" if exists else "❌"
        print(f"    {status} {filename:30s} - {description}")
        if not exists:
            all_exist = False
    
    # Step 5: Display file contents summary
    print("\n" + "=" * 70)
    print("File Contents Summary")
    print("=" * 70)
    
    # Config summary
    config_path = workspace_path / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
            print(f"\n  📄 config.json:")
            print(f"    • Swarm name: {config.get('name', 'N/A')}")
            print(f"    • Max loops: {config.get('max_loops', 'N/A')}")
            print(f"    • Autosave enabled: {config.get('autosave', False)}")
            if "_autosave_metadata" in config:
                meta = config["_autosave_metadata"]
                print(f"    • Saved at: {meta.get('timestamp', 'N/A')}")
    
    # State summary
    state_path = workspace_path / "state.json"
    if state_path.exists():
        with open(state_path, "r") as f:
            state = json.load(f)
            print(f"\n  📄 state.json:")
            print(f"    • Swarm name: {state.get('swarm_name', 'N/A')}")
            print(f"    • Timestamp: {state.get('timestamp', 'N/A')}")
    
    # Metadata summary
    metadata_path = workspace_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            print(f"\n  📄 metadata.json:")
            print(f"    • Status: {metadata.get('status', 'N/A')}")
            print(f"    • Agents count: {metadata.get('agents_count', 'N/A')}")
            print(f"    • Loops completed: {metadata.get('loops_completed', 'N/A')}")
    
    # Conversation history summary
    conversation_path = workspace_path / "conversation_history.json"
    if conversation_path.exists():
        with open(conversation_path, "r") as f:
            conversation = json.load(f)
            print(f"\n  📄 conversation_history.json:")
            if isinstance(conversation, list):
                print(f"    • Total messages: {len(conversation)}")
                if len(conversation) > 0:
                    print(f"    • First message role: {conversation[0].get('role', 'N/A')}")
                    print(f"    • Last message role: {conversation[-1].get('role', 'N/A')}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    if all_exist:
        print("  ✅ All autosave files were created successfully!")
        print(f"\n  📁 Files location: {workspace_path}")
        print("\n  💡 You can manually inspect the files to see full contents.")
    else:
        print("  ⚠️  Some files are missing. Check the output above.")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Ensure WORKSPACE_DIR is set
    if not os.getenv("WORKSPACE_DIR"):
        os.environ["WORKSPACE_DIR"] = "agent_workspace"
        print("ℹ️  Set WORKSPACE_DIR to 'agent_workspace'")
    
    main()
