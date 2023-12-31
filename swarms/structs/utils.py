from typing import Dict, Any, List
from swarms.structs.agent import Agent


# Helper functions for manager/corporate agents
def parse_tasks(
    task: str = None,
) -> Dict[str, Any]:
    """Parse tasks

    Args:
        task (str, optional): _description_. Defaults to None.

    Returns:
        Dict[str, Any]: _description_
    """
    tasks = {}
    for line in task.split("\n"):
        if line.startswith("<agent_id>") and line.endwith(
            "</agent_id>"
        ):
            agent_id, task = line[10:-11].split("><")
            tasks[agent_id] = task
    return tasks


def find_agent_by_id(
    agent_id: str = None, agents: List[Agent] = None, *args, **kwargs
) -> Agent:
    """Find agent by id

    Args:
        agent_id (str, optional): _description_. Defaults to None.
        agents (List[Agent], optional): _description_. Defaults to None.

    Returns:
        Agent: _description_
    """
    for agent in agents:
        if agent.id == agent_id:
            return agent
    return None


def distribute_tasks(
    task: str = None, agents: List[Agent] = None, *args, **kwargs
):
    """Distribute tasks to agents

    Args:
        task (str, optional): _description_. Defaults to None.
        agents (List[Agent], optional): _description_. Defaults to None.
    """
    # Parse the task to extract tasks and agent id
    tasks = parse_tasks(task)

    # Distribute tasks to agents
    for agent_id, task in tasks.item():
        assigned_agent = find_agent_by_id(agent_id, agents)
        if assigned_agent:
            print(f"Assigning task {task} to agent {agent_id}")
            output = assigned_agent.run(task, *args, **kwargs)
            print(f"Output from agent {agent_id}: {output}")
        else:
            print(
                f"No agent found with ID {agent_id}. Task '{task}' is"
                " not assigned."
            )
