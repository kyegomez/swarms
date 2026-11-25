from typing import List, Union, Dict, Any


from swarms.structs.context_utils import (
    run_on_conversation as _run_on_conversation,
)
from swarms.structs.omni_agent_types import AgentListType
from swarms.utils.loguru_logger import initialize_logger
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.output_types import OutputType

logger = initialize_logger(log_folder="swarming_architectures")


def circular_swarm(
    agents: AgentListType,
    tasks: List[str],
    output_type: OutputType = "dict",
) -> Union[Dict[str, Any], List[str]]:
    """
    Implements a circular swarm where agents pass tasks in a circular manner.

    Args:
        agents (AgentListType): A list of Agent objects to participate in the swarm.
        tasks (List[str]): A list of tasks to be processed by the agents.
        output_type (OutputType, optional): The format of the output. Defaults to "dict".

    Returns:
        Union[Dict[str, Any], List[str]]: The formatted output of the swarm's processing.
            If output_type is "dict", returns a dictionary containing the conversation history.
            If output_type is "list", returns a list of responses.

    Raises:
        ValueError: If agents or tasks lists are empty.
    """

    # Ensure agents is a flat list of Agent objects
    flat_agents = (
        [agent for sublist in agents for agent in sublist]
        if isinstance(agents[0], list)
        else agents
    )

    if not flat_agents or not tasks:
        raise ValueError("Agents and tasks lists cannot be empty.")

    conversation = Conversation()

    for task in tasks:
        # Once per task, not once per agent: every agent reads the same turn.
        conversation.add(
            role="User",
            content=task,
        )

        for agent in flat_agents:
            _run_on_conversation(agent, conversation)

    return history_output_formatter(conversation, output_type)


def grid_swarm(
    agents: AgentListType,
    tasks: List[str],
    output_type: OutputType = "dict",
) -> Union[Dict[str, Any], List[str]]:
    """
    Implements a grid swarm where agents are arranged in a square grid pattern.

    Args:
        agents (AgentListType): A list of Agent objects to participate in the swarm.
        tasks (List[str]): A list of tasks to be processed by the agents.
        output_type (OutputType, optional): The format of the output. Defaults to "dict".

    Returns:
        Union[Dict[str, Any], List[str]]: The formatted output of the swarm's processing.
            If output_type is "dict", returns a dictionary containing the conversation history.
            If output_type is "list", returns a list of responses.

    Raises:
        ValueError: If agents or tasks lists are empty.
    """
    if not agents or not tasks:
        raise ValueError("Agents and tasks lists cannot be empty.")

    conversation = Conversation()

    conversation.add(
        role="User",
        content=tasks,
    )

    for index, task in enumerate(tasks):
        agent = agents[index % len(agents)]
        response = agent.run(task)
        conversation.add(
            role=agent.agent_name,
            content=response,
        )

    return history_output_formatter(conversation, output_type)


# One-to-One Communication between two agents
