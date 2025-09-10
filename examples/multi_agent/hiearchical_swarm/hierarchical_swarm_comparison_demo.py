#!/usr/bin/env python3
"""
Hierarchical Swarm Comparison Demo

This demo compares traditional swarm execution (without streaming)
versus streaming execution to show the difference in behavior.
"""

from swarms.structs.hiearchical_swarm import HierarchicalSwarm
from swarms.agents import Agent


def create_agents():
    """Create specialized agents for the swarm."""
    return [
        Agent(
            agent_name="Research_Agent",
            agent_description="Specialized in gathering and analyzing information",
            system_prompt="You are a research specialist. Provide detailed, accurate information on any topic.",
            model_name="gpt-4o-mini",
            max_loops=1,
        ),
        Agent(
            agent_name="Analysis_Agent",
            agent_description="Expert at analyzing data and drawing insights",
            system_prompt="You are an analysis expert. Break down complex information and provide clear insights.",
            model_name="gpt-4o-mini",
            max_loops=1,
        ),
        Agent(
            agent_name="Summary_Agent",
            agent_description="Skilled at creating concise summaries",
            system_prompt="You are a summarization expert. Create clear, concise summaries of complex topics.",
            model_name="gpt-4o-mini",
            max_loops=1,
        ),
    ]


def run_traditional_swarm():
    """Run swarm without streaming callbacks."""
    print("🔇 TRADITIONAL SWARM EXECUTION (No Streaming)")
    print("-" * 50)

    agents = create_agents()
    swarm = HierarchicalSwarm(
        name="Traditional_Swarm",
        description="Traditional swarm execution",
        agents=agents,
        max_loops=1,
        verbose=False,
        director_model_name="gpt-4o-mini",
    )

    task = "What are the main benefits of renewable energy sources?"

    print(f"Task: {task}")

    result = swarm.run(task=task)

    print("\nResult:")
    if isinstance(result, dict):
        for key, value in result.items():
            print(
                f"{key}: {value[:200]}..."
                if len(str(value)) > 200
                else f"{key}: {value}"
            )
    else:
        print(
            result[:500] + "..." if len(str(result)) > 500 else result
        )


def run_streaming_swarm():
    """Run swarm with streaming callbacks."""

    def simple_callback(agent_name: str, chunk: str, is_final: bool):
        if chunk.strip():
            if is_final:
                print(f"\n✅ {agent_name} completed")
            else:
                print(
                    f"🔄 {agent_name}: {chunk[:50]}..."
                    if len(chunk) > 50
                    else f"🔄 {agent_name}: {chunk}"
                )

    print("\n🎯 STREAMING SWARM EXECUTION")
    print("-" * 50)

    agents = create_agents()
    swarm = HierarchicalSwarm(
        name="Streaming_Swarm",
        description="Swarm with streaming callbacks",
        agents=agents,
        max_loops=1,
        verbose=False,
        director_model_name="gpt-4o-mini",
    )

    task = "What are the main benefits of renewable energy sources?"

    print(f"Task: {task}")

    result = swarm.run(task=task, streaming_callback=simple_callback)

    print("\nResult:")
    if isinstance(result, dict):
        for key, value in result.items():
            print(
                f"{key}: {value[:200]}..."
                if len(str(value)) > 200
                else f"{key}: {value}"
            )
    else:
        print(
            result[:500] + "..." if len(str(result)) > 500 else result
        )


if __name__ == "__main__":
    print("🔄 HIERARCHICAL SWARM COMPARISON DEMO")
    print("=" * 50)
    print("Comparing traditional vs streaming execution\n")

    # Run traditional first
    run_traditional_swarm()

    # Run streaming second
    run_streaming_swarm()

    print("\n" + "=" * 50)
    print("✨ Comparison complete!")
    print("Notice how streaming shows progress in real-time")
