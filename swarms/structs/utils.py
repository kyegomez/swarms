from typing import List
from swarms.structs.agent import Agent
from swarms.structs.ma_blocks import (
    find_agent_by_id as find_agent_by_id_ma_blocks,
)


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
        agent = find_agent_by_id_ma_blocks(agents, agent_id)

        if agent is None:
            raise ValueError(f"Agent with ID {agent_id} not found")

        if task:
            return agent.run(task, *args, **kwargs)
        else:
            return agent
    except ValueError as e:
        raise ValueError(f"Error finding agent by ID: {str(e)}")
