from typing import List
from swarms.structs.agent import Agent


def find_agent_by_id(
    agent_id: str = None,
    agents: List[Agent] = None,
    task: str = None,
    *args,
    **kwargs,
) -> Agent:
    """Find agent by id

    Args:
        agent_id (str, optional): _description_. Defaults to None.
        agents (List[Agent], optional): _description_. Defaults to None.

    Returns:
        Agent: _description_
    """
    try:
        print(f"Searching for agent with ID: {agent_id}")
        for agent in agents:
            if agent.id == agent_id:
                print(f"Found agent with ID {agent_id}")
                if task:
                    print(f"Running task: {task}")
                    return agent.run(task, *args, **kwargs)
                else:
                    return agent
        print(f"No agent found with ID {agent_id}")
        return None
    except Exception as e:
        print(f"Error finding agent by ID: {str(e)}")
        return None


def find_agent_by_name(
    agent_name: str = None,
    agents: List[Agent] = None,
    task: str = None,
    *args,
    **kwargs,
) -> Agent:
    """Find agent by name

    Args:
        agent_name (str): _description_
        agents (List[Agent]): _description_

    Returns:
        Agent: _description_
    """
    try:
        print(f"Searching for agent with name: {agent_name}")
        for agent in agents:
            if agent.name == agent_name:
                print(f"Found agent with name {agent_name}")
                if task:
                    print(f"Running task: {task}")
                    return agent.run(task, *args, **kwargs)
                else:
                    return agent
        print(f"No agent found with name {agent_name}")
        return None
    except Exception as e:
        print(f"Error finding agent by name: {str(e)}")
        return None
