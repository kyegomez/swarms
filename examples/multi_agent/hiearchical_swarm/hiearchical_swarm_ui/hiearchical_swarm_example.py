"""
Hierarchical Swarm with Arasaka Dashboard Example

This example demonstrates the new interactive dashboard functionality for the
hierarchical swarm, featuring a futuristic Arasaka Corporation-style interface
with red and black color scheme.
"""

from swarms.structs.hiearchical_swarm import HierarchicalSwarm
from swarms.structs.agent import Agent


def main():
    """
    Demonstrate the hierarchical swarm with interactive dashboard.
    """
    print("🚀 Initializing Swarms Corporation Hierarchical Swarm...")

    # Create specialized agents
    research_agent = Agent(
        agent_name="Research-Analyst",
        agent_description="Specialized in comprehensive research and data gathering",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
    )

    analysis_agent = Agent(
        agent_name="Data-Analyst",
        agent_description="Expert in data analysis and pattern recognition",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
    )

    strategy_agent = Agent(
        agent_name="Strategy-Consultant",
        agent_description="Specialized in strategic planning and recommendations",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
    )

    # Create hierarchical swarm with interactive dashboard
    swarm = HierarchicalSwarm(
        name="Swarms Corporation Operations",
        description="Enterprise-grade hierarchical swarm for complex task execution",
        agents=[research_agent, analysis_agent, strategy_agent],
        max_loops=2,
        interactive=True,  # Enable the Arasaka dashboard
        verbose=True,
    )

    print("\n🎯 Swarm initialized successfully!")
    print(
        "📊 Interactive dashboard will be displayed during execution."
    )
    print(
        "💡 The swarm will prompt you for a task when you call swarm.run()"
    )

    # Run the swarm (task will be prompted interactively)
    result = swarm.run()

    print("\n✅ Swarm execution completed!")
    print("📋 Final result:")
    print(result)


if __name__ == "__main__":
    main()
