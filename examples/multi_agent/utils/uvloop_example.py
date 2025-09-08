"""
Example demonstrating the use of uvloop for running multiple agents concurrently.

This example shows how to use the new uvloop-based functions:
- run_agents_concurrently_uvloop: For running multiple agents with the same task
- run_agents_with_tasks_uvloop: For running agents with different tasks

uvloop provides significant performance improvements over standard asyncio,
especially for I/O-bound operations and concurrent task execution.
"""

import os
from swarms.structs.multi_agent_exec import (
    run_agents_concurrently_uvloop,
    run_agents_with_tasks_uvloop,
)
from swarms.structs.agent import Agent


def create_example_agents(num_agents: int = 3):
    """Create example agents for demonstration."""
    agents = []
    for i in range(num_agents):
        agent = Agent(
            agent_name=f"Agent_{i+1}",
            system_prompt=f"You are Agent {i+1}, a helpful AI assistant.",
            model_name="gpt-4o-mini",  # Using a lightweight model for examples
            max_loops=1,
            autosave=False,
            verbose=False,
        )
        agents.append(agent)
    return agents


def example_same_task():
    """Example: Running multiple agents with the same task using uvloop."""
    print("=== Example 1: Same Task for All Agents (uvloop) ===")

    agents = create_example_agents(3)
    task = (
        "Write a one-sentence summary about artificial intelligence."
    )

    print(f"Running {len(agents)} agents with the same task...")
    print(f"Task: {task}")

    try:
        results = run_agents_concurrently_uvloop(agents, task)

        print("\nResults:")
        for i, result in enumerate(results, 1):
            print(f"Agent {i}: {result}")

    except Exception as e:
        print(f"Error: {e}")


def example_different_tasks():
    """Example: Running agents with different tasks using uvloop."""
    print(
        "\n=== Example 2: Different Tasks for Each Agent (uvloop) ==="
    )

    agents = create_example_agents(3)
    tasks = [
        "Explain what machine learning is in simple terms.",
        "Describe the benefits of cloud computing.",
        "What are the main challenges in natural language processing?",
    ]

    print(f"Running {len(agents)} agents with different tasks...")

    try:
        results = run_agents_with_tasks_uvloop(agents, tasks)

        print("\nResults:")
        for i, (result, task) in enumerate(zip(results, tasks), 1):
            print(f"Agent {i} (Task: {task[:50]}...):")
            print(f"  Response: {result}")
            print()

    except Exception as e:
        print(f"Error: {e}")


def performance_comparison():
    """Demonstrate the performance benefit of uvloop vs standard asyncio."""
    print("\n=== Performance Comparison ===")

    # Note: This is a conceptual example. In practice, you'd need to measure actual performance
    print("uvloop vs Standard asyncio:")
    print("• uvloop: Cython-based event loop, ~2-4x faster")
    print("• Better for I/O-bound operations")
    print("• Lower latency and higher throughput")
    print("• Especially beneficial for concurrent agent execution")
    print("• Automatic fallback to asyncio if uvloop unavailable")


if __name__ == "__main__":
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Please set your OPENAI_API_KEY environment variable to run this example."
        )
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        exit(1)

    print("🚀 uvloop Multi-Agent Execution Examples")
    print("=" * 50)

    # Run examples
    example_same_task()
    example_different_tasks()
    performance_comparison()

    print("\n✅ Examples completed!")
    print("\nTo use uvloop functions in your code:")
    print(
        "from swarms.structs.multi_agent_exec import run_agents_concurrently_uvloop"
    )
    print("results = run_agents_concurrently_uvloop(agents, task)")
