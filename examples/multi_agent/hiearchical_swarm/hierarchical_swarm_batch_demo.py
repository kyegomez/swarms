#!/usr/bin/env python3
"""
Hierarchical Swarm Batch Processing Demo

This demo shows how to use streaming callbacks with batch processing
to handle multiple tasks sequentially with real-time feedback.
"""

import time
from typing import Callable
from swarms.structs.hiearchical_swarm import HierarchicalSwarm
from swarms.agents import Agent


def create_batch_callback() -> Callable[[str, str, bool], None]:
    """Create a callback optimized for batch processing."""

    def batch_callback(agent_name: str, chunk: str, is_final: bool):
        timestamp = time.strftime("%H:%M:%S")

        if chunk.strip():
            if is_final:
                print(f"\n✅ [{timestamp}] {agent_name} COMPLETED")
            else:
                # Shorter output for batch processing
                print(f"🔄 {agent_name}: {chunk[:30]}..." if len(chunk) > 30 else f"🔄 {agent_name}: {chunk}")

    return batch_callback


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
    ]


if __name__ == "__main__":
    print("📦 HIERARCHICAL SWARM BATCH PROCESSING DEMO")
    print("="*50)

    # Create agents and swarm
    agents = create_agents()
    swarm = HierarchicalSwarm(
        name="Batch_Processing_Swarm",
        description="Swarm for batch processing multiple tasks",
        agents=agents,
        max_loops=1,
        verbose=False,  # Reduce verbosity for cleaner batch output
        director_model_name="gpt-4o-mini",
    )

    # Define multiple tasks
    tasks = [
        "What are the environmental benefits of solar energy?",
        "How does wind power contribute to sustainable development?",
        "What are the economic advantages of hydroelectric power?"
    ]

    print(f"Processing {len(tasks)} tasks:")
    for i, task in enumerate(tasks, 1):
        print(f"  {i}. {task}")
    print()

    # Create streaming callback
    streaming_callback = create_batch_callback()

    print("🎬 EXECUTING BATCH WITH STREAMING CALLBACKS...")
    print("Each task will show real-time progress:\n")

    # Execute batch with streaming
    results = swarm.batched_run(
        tasks=tasks,
        streaming_callback=streaming_callback,
    )

    print("\n" + "="*50)
    print("🎉 BATCH PROCESSING COMPLETED!")
    print(f"Processed {len(results)} tasks")

    # Show summary
    print("\n📊 SUMMARY:")
    for i, result in enumerate(results, 1):
        print(f"  Task {i}: {'Completed' if result else 'Failed'}")
