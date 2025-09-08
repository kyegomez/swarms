"""
Example demonstrating running agents with different tasks using uvloop.

This example shows how to use run_agents_with_tasks_uvloop to execute
different tasks across multiple agents concurrently.
"""

import os

from swarms.structs.agent import Agent
from swarms.structs.multi_agent_exec import (
    run_agents_with_tasks_uvloop,
)


def create_example_agents(num_agents: int = 3):
    """
    Create example agents for demonstration.

    Args:
        num_agents: Number of agents to create

    Returns:
        List of Agent instances
    """
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


def run_different_tasks_example():
    """
    Run agents with different tasks using uvloop.

    Returns:
        List of results from each agent
    """
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set"
        )

    agents = create_example_agents(3)
    tasks = [
        "Explain what machine learning is in simple terms.",
        "Describe the benefits of cloud computing.",
        "What are the main challenges in natural language processing?",
    ]

    results = run_agents_with_tasks_uvloop(agents, tasks)
    return results


if __name__ == "__main__":
    results = run_different_tasks_example()
    # Results can be processed further as needed
