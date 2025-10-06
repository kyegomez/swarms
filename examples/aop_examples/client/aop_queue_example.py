#!/usr/bin/env python3
"""
Example demonstrating the AOP queue system for agent execution.

This example shows how to use the new queue-based execution system
in the AOP framework for improved performance and reliability.
"""

import time
from swarms import Agent
from swarms.structs.aop import AOP


def main():
    """Demonstrate AOP queue functionality."""

    # Create some sample agents
    agent1 = Agent(
        agent_name="Research Agent",
        agent_description="Specialized in research tasks",
        model_name="gpt-4",
        max_loops=1,
    )

    agent2 = Agent(
        agent_name="Writing Agent",
        agent_description="Specialized in writing tasks",
        model_name="gpt-4",
        max_loops=1,
    )

    # Create AOP with queue enabled
    aop = AOP(
        server_name="Queue Demo Cluster",
        description="A demonstration of queue-based agent execution",
        queue_enabled=True,
        max_workers_per_agent=2,  # 2 workers per agent
        max_queue_size_per_agent=100,  # Max 100 tasks per queue
        processing_timeout=60,  # 60 second timeout
        retry_delay=2.0,  # 2 second delay between retries
        verbose=True,
    )

    # Add agents to the cluster
    print("Adding agents to cluster...")
    aop.add_agent(agent1, tool_name="researcher")
    aop.add_agent(agent2, tool_name="writer")

    # Get initial queue stats
    print("\nInitial queue stats:")
    stats = aop.get_queue_stats()
    print(f"Stats: {stats}")

    # Add some tasks to the queues
    print("\nAdding tasks to queues...")

    # Add high priority research task
    research_task_id = aop.task_queues["researcher"].add_task(
        task="Research the latest developments in quantum computing",
        priority=10,  # High priority
        max_retries=2,
    )
    print(f"Added research task: {research_task_id}")

    # Add medium priority writing task
    writing_task_id = aop.task_queues["writer"].add_task(
        task="Write a summary of AI trends in 2024",
        priority=5,  # Medium priority
        max_retries=3,
    )
    print(f"Added writing task: {writing_task_id}")

    # Add multiple low priority tasks
    for i in range(3):
        task_id = aop.task_queues["researcher"].add_task(
            task=f"Research task {i+1}: Analyze market trends",
            priority=1,  # Low priority
            max_retries=1,
        )
        print(f"Added research task {i+1}: {task_id}")

    # Get updated queue stats
    print("\nUpdated queue stats:")
    stats = aop.get_queue_stats()
    print(f"Stats: {stats}")

    # Monitor task progress
    print("\nMonitoring task progress...")
    for _ in range(10):  # Monitor for 10 iterations
        time.sleep(1)

        # Check research task status
        research_status = aop.get_task_status(
            "researcher", research_task_id
        )
        print(
            f"Research task status: {research_status['task']['status'] if research_status['success'] else 'Error'}"
        )

        # Check writing task status
        writing_status = aop.get_task_status(
            "writer", writing_task_id
        )
        print(
            f"Writing task status: {writing_status['task']['status'] if writing_status['success'] else 'Error'}"
        )

        # Get current queue stats
        current_stats = aop.get_queue_stats()
        if current_stats["success"]:
            for agent_name, agent_stats in current_stats[
                "stats"
            ].items():
                print(
                    f"{agent_name}: {agent_stats['pending_tasks']} pending, {agent_stats['processing_tasks']} processing, {agent_stats['completed_tasks']} completed"
                )

        print("---")

    # Demonstrate queue management
    print("\nDemonstrating queue management...")

    # Pause the research agent queue
    print("Pausing research agent queue...")
    aop.pause_agent_queue("researcher")

    # Get queue status
    research_queue_status = aop.task_queues["researcher"].get_status()
    print(f"Research queue status: {research_queue_status.value}")

    # Resume the research agent queue
    print("Resuming research agent queue...")
    aop.resume_agent_queue("researcher")

    # Clear all queues
    print("Clearing all queues...")
    cleared = aop.clear_all_queues()
    print(f"Cleared tasks: {cleared}")

    # Final stats
    print("\nFinal queue stats:")
    final_stats = aop.get_queue_stats()
    print(f"Final stats: {final_stats}")

    print("\nQueue demonstration completed!")


if __name__ == "__main__":
    main()
