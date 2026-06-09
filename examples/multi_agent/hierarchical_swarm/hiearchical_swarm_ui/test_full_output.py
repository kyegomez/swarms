"""
Test script for full agent output display in the Hierarchical Swarms Dashboard.
"""

from swarms.structs.hiearchical_swarm import HierarchicalSwarm
from swarms.structs.agent import Agent


def test_full_output():
    """Test the full output display functionality."""

    print("🔍 Testing full agent output display...")

    # Create agents that will produce substantial output
    agent1 = Agent(
        agent_name="Research-Agent",
        agent_description="A research agent that produces detailed output",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
    )

    agent2 = Agent(
        agent_name="Analysis-Agent",
        agent_description="An analysis agent that provides comprehensive analysis",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
    )

    # Create swarm with dashboard and detailed view enabled
    swarm = HierarchicalSwarm(
        name="Full Output Test Swarm",
        description="A test swarm for verifying full agent output display",
        agents=[agent1, agent2],
        max_loops=1,
        interactive=True,
        verbose=True,
    )

    print("✅ Created swarm with detailed view enabled")
    print(
        "📊 Dashboard should show full agent outputs without truncation"
    )

    # Run with a task that will generate substantial output
    swarm.run(
        task="Provide a comprehensive analysis of artificial intelligence trends in 2024, including detailed explanations of each trend"
    )

    print("\n✅ Test completed!")
    print("📋 Check the dashboard for full agent outputs")


if __name__ == "__main__":
    test_full_output()
