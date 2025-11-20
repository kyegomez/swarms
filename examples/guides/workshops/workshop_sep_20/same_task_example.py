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


agents = create_example_agents(3)

task = "Write a creative story about sci fi with no cliches. Make it 1000 words."

results = run_agents_concurrently_uvloop(agents=agents, task=task)


print(results)
