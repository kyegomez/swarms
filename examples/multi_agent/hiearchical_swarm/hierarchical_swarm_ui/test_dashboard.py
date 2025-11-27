"""
Test script for the Arasaka Dashboard functionality.
"""

from swarms.structs.hierarchical_swarm import HierarchicalSwarm
from swarms.structs.agent import Agent


def test_dashboard():
    """Test the dashboard functionality with a simple task."""

    # Create simple agents
    agent1 = Agent(
        agent_name="Test-Agent-1",
        agent_description="A test agent for dashboard verification",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
    )

    agent2 = Agent(
        agent_name="Test-Agent-2",
        agent_description="Another test agent for dashboard verification",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
    )

    # Create swarm with dashboard
    swarm = HierarchicalSwarm(
        name="Dashboard Test Swarm",
        agents=[agent1, agent2],
        max_loops=1,
        interactive=True,
        verbose=True,
    )

    print("🧪 Testing Arasaka Dashboard...")
    print("📊 Dashboard should appear and prompt for task input")

    # Run with a simple task
    result = swarm.run(
        task="Create a simple summary of artificial intelligence trends"
    )

    print("\n✅ Test completed!")
    print("📋 Result preview:")
    print(
        str(result)[:500] + "..."
        if len(str(result)) > 500
        else str(result)
    )


if __name__ == "__main__":
    test_dashboard()
