import os

from swarms.structs.agent import Agent
from swarms.structs.multi_agent_exec import (
    run_agents_concurrently_uvloop,
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


def run_same_task_example():
    """
    Run multiple agents with the same task using uvloop.

    Returns:
        List of results from each agent
    """
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set"
        )

    agents = create_example_agents(3)
    task = (
        "Write a one-sentence summary about artificial intelligence."
    )

    results = run_agents_concurrently_uvloop(agents, task)
    return results


if __name__ == "__main__":
    results = run_same_task_example()
    # Results can be processed further as needed
