"""
Debug script for the Arasaka Dashboard to test agent output display.
"""

from swarms.structs.hierarchical_swarm import HierarchicalSwarm
from swarms.structs.agent import Agent


def debug_dashboard():
    """Debug the dashboard functionality."""

    print("🔍 Starting dashboard debug...")

    # Create simple agents with clear names
    agent1 = Agent(
        agent_name="Research-Agent",
        agent_description="A research agent for testing",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
    )

    agent2 = Agent(
        agent_name="Analysis-Agent",
        agent_description="An analysis agent for testing",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
    )

    print(
        f"✅ Created agents: {agent1.agent_name}, {agent2.agent_name}"
    )

    # Create swarm with dashboard
    swarm = HierarchicalSwarm(
        name="Debug Swarm",
        description="A test swarm for debugging dashboard functionality",
        agents=[agent1, agent2],
        max_loops=1,
        interactive=True,
        verbose=True,
    )

    print("✅ Created swarm with dashboard")
    print("📊 Dashboard should now show agents in PENDING status")

    # Wait a moment to see the initial dashboard
    import time

    time.sleep(3)

    print("\n🚀 Starting swarm execution...")

    # Run with a simple task
    result = swarm.run(
        task="Create a brief summary of machine learning"
    )

    print("\n✅ Debug completed!")
    print("📋 Final result preview:")
    print(
        str(result)[:300] + "..."
        if len(str(result)) > 300
        else str(result)
    )


if __name__ == "__main__":
    debug_dashboard()
