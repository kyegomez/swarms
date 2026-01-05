"""
Test script for multi-loop agent tracking in the Hierarchical Swarms Dashboard.
"""

from swarms.structs.hiearchical_swarm import HierarchicalSwarm
from swarms.structs.agent import Agent


def test_multi_loop():
    """Test the multi-loop agent tracking functionality."""

    print("🔍 Testing multi-loop agent tracking...")

    # Create agents
    agent1 = Agent(
        agent_name="Research-Agent",
        agent_description="A research agent for multi-loop testing",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
    )

    agent2 = Agent(
        agent_name="Analysis-Agent",
        agent_description="An analysis agent for multi-loop testing",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
    )

    # Create swarm with multiple loops
    swarm = HierarchicalSwarm(
        name="Multi-Loop Test Swarm",
        description="A test swarm for verifying multi-loop agent tracking",
        agents=[agent1, agent2],
        max_loops=3,  # Multiple loops to test history tracking
        interactive=True,
        verbose=True,
    )

    print("✅ Created swarm with multi-loop tracking")
    print(
        "📊 Dashboard should show agent outputs across multiple loops"
    )
    print("🔄 Each loop will add new rows to the monitoring matrix")

    # Run with a task that will benefit from multiple iterations
    swarm.run(
        task="Analyze the impact of artificial intelligence on healthcare, then refine the analysis with additional insights, and finally provide actionable recommendations"
    )

    print("\n✅ Multi-loop test completed!")
    print("📋 Check the dashboard for agent outputs across all loops")


if __name__ == "__main__":
    test_multi_loop()
