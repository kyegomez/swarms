#!/usr/bin/env python3
"""
Basic Hierarchical Swarm Streaming Demo

Minimal example showing the core streaming callback functionality.
"""

from swarms.structs.hierarchical_swarm import HierarchicalSwarm
from swarms.agents import Agent


def simple_callback(agent_name: str, chunk: str, is_final: bool):
    """Simple callback that shows agent progress."""
    if chunk.strip():
        if is_final:
            print(f"✅ {agent_name} finished")
        else:
            print(f"🔄 {agent_name}: {chunk}")


if __name__ == "__main__":
    print("🎯 BASIC HIERARCHICAL SWARM STREAMING")

    # Create a simple agent
    agent = Agent(
        agent_name="Simple_Agent",
        agent_description="A simple agent for demonstration",
        system_prompt="You are a helpful assistant.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    # Create swarm
    swarm = HierarchicalSwarm(
        name="Basic_Swarm",
        description="Basic streaming demo",
        agents=[agent],
        max_loops=1,
        director_model_name="gpt-4o-mini",
    )

    # Simple task
    task = "Explain what artificial intelligence is in simple terms."

    print(f"Task: {task}")
    print("\nExecuting with streaming callback:\n")

    # Run with streaming
    result = swarm.run(task=task, streaming_callback=simple_callback)

    print("\n" + "=" * 30)
    print("Final result:")
    print(result)
