import math
from typing import List, Union, Dict, Any


from swarms.structs.agent import Agent
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
        for agent in flat_agents:
            conversation.add(
                role="User",
                message=task,
            )
            response = agent.run(conversation.get_str())
            conversation.add(
                role=agent.agent_name,
                message=response,
            )

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
    conversation = Conversation()

    conversation.add(
        role="User",
        message=tasks,
    )

    grid_size = int(
        len(agents) ** 0.5
    )  # Assuming agents can form a perfect square grid
    for i in range(grid_size):
        for j in range(grid_size):
            if tasks:
                task = tasks.pop(0)
                response = agents[i * grid_size + j].run(task)
                conversation.add(
                    role=agents[i * grid_size + j].agent_name,
                    message=response,
                )

    return history_output_formatter(conversation, output_type)


# Linear Swarm: Agents process tasks in a sequential linear manner
def linear_swarm(
    agents: AgentListType,
    tasks: List[str],
    output_type: OutputType = "dict",
) -> Union[Dict[str, Any], List[str]]:
    """
    Implements a linear swarm where agents process tasks in a sequential manner.

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

    for agent in agents:
        if tasks:
            task = tasks.pop(0)
            conversation.add(
                role="User",
                message=task,
            )
            response = agent.run(conversation.get_str())
            conversation.add(
                role=agent.agent_name,
                message=response,
            )

    return history_output_formatter(conversation, output_type)


# Star Swarm: A central agent first processes all tasks, followed by others
def star_swarm(
    agents: AgentListType,
    tasks: List[str],
    output_type: OutputType = "dict",
) -> Union[Dict[str, Any], List[str]]:
    """
    Implements a star swarm where a central agent processes tasks first, followed by others.

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
    center_agent = agents[0]  # The central agent

    for task in tasks:
        # Central agent processes the task
        conversation.add(
            role="User",
            message=task,
        )
        center_response = center_agent.run(conversation.get_str())
        conversation.add(
            role=center_agent.agent_name,
            message=center_response,
        )

        # Other agents process the same task
        for agent in agents[1:]:
            response = agent.run(task)
            conversation.add(
                role=agent.agent_name,
                message=response,
            )

    return history_output_formatter(conversation, output_type)


# Mesh Swarm: Agents work on tasks randomly from a task queue until all tasks are processed
def mesh_swarm(
    agents: AgentListType,
    tasks: List[str],
    output_type: OutputType = "dict",
) -> Union[Dict[str, Any], List[str]]:
    """
    Implements a mesh swarm where agents work on tasks randomly from a task queue.

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
        message=tasks,
    )
    task_queue = tasks.copy()

    while task_queue:
        for agent in agents:
            if task_queue:
                task = task_queue.pop(0)
                response = agent.run(task)
                conversation.add(
                    role=agent.agent_name,
                    message=response,
                )

    return history_output_formatter(conversation, output_type)


# Pyramid Swarm: Agents are arranged in a pyramid structure
def pyramid_swarm(
    agents: AgentListType,
    tasks: List[str],
    output_type: OutputType = "dict",
) -> Union[Dict[str, Any], List[str]]:
    """
    Implements a pyramid swarm where agents are arranged in a pyramid structure.

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

    levels = int(
        (-1 + (1 + 8 * len(agents)) ** 0.5) / 2
    )  # Number of levels in the pyramid

    for i in range(levels):
        for j in range(i + 1):
            if tasks:
                task = tasks.pop(0)
                agent_index = int(i * (i + 1) / 2 + j)
                response = agents[agent_index].run(task)
                conversation.add(
                    role=agents[agent_index].agent_name,
                    message=response,
                )

    return history_output_formatter(conversation, output_type)


def fibonacci_swarm(
    agents: AgentListType,
    tasks: List[str],
    output_type: OutputType = "dict",
) -> Union[Dict[str, Any], List[str]]:
    """
    Implements a fibonacci swarm where agents are selected based on the fibonacci sequence.

    Args:
        agents (AgentListType): A list of Agent objects to participate in the swarm.
        tasks (List[str]): A list of tasks to be processed by the agents.
        output_type (OutputType, optional): The format of the output. Defaults to "dict".

    Returns:
        Union[Dict[str, Any], List[str]]: The formatted output of the swarm's processing.
            If output_type is "dict", returns a dictionary containing the conversation history.
            If output_type is "list", returns a list of responses.
    """
    conversation = Conversation()
    conversation.add(
        role="User",
        message=tasks,
    )
    fib = [1, 1]
    while len(fib) < len(agents):
        fib.append(fib[-1] + fib[-2])
    for i in range(len(fib)):
        for j in range(fib[i]):
            if tasks:
                task = tasks.pop(0)
                response = agents[int(sum(fib[:i]) + j)].run(task)
                conversation.add(
                    role=agents[int(sum(fib[:i]) + j)].agent_name,
                    message=response,
                )

    return history_output_formatter(conversation, output_type)


def prime_swarm(
    agents: AgentListType,
    tasks: List[str],
    output_type: OutputType = "dict",
) -> Union[Dict[str, Any], List[str]]:
    """
    Implements a prime swarm where agents are selected based on prime numbers.

    Args:
        agents (AgentListType): A list of Agent objects to participate in the swarm.
        tasks (List[str]): A list of tasks to be processed by the agents.
        output_type (OutputType, optional): The format of the output. Defaults to "dict".

    Returns:
        Union[Dict[str, Any], List[str]]: The formatted output of the swarm's processing.
            If output_type is "dict", returns a dictionary containing the conversation history.
            If output_type is "list", returns a list of responses.
    """
    conversation = Conversation()
    conversation.add(
        role="User",
        message=tasks,
    )
    primes = [
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
    ]  # First 25 prime numbers
    for prime in primes:
        if prime < len(agents) and tasks:
            task = tasks.pop(0)
            output = agents[prime].run(task)
            conversation.add(
                role=agents[prime].agent_name,
                message=output,
            )
    return history_output_formatter(conversation, output_type)


def power_swarm(
    agents: AgentListType,
    tasks: List[str],
    output_type: OutputType = "dict",
) -> Union[Dict[str, Any], List[str]]:
    """
    Implements a power swarm where agents are selected based on powers of 2.

    Args:
        agents (AgentListType): A list of Agent objects to participate in the swarm.
        tasks (List[str]): A list of tasks to be processed by the agents.
        output_type (OutputType, optional): The format of the output. Defaults to "dict".

    Returns:
        Union[Dict[str, Any], List[str]]: The formatted output of the swarm's processing.
            If output_type is "dict", returns a dictionary containing the conversation history.
            If output_type is "list", returns a list of responses.
    """
    conversation = Conversation()
    conversation.add(
        role="User",
        message=tasks,
    )
    powers = [2**i for i in range(int(len(agents) ** 0.5))]
    for power in powers:
        if power < len(agents) and tasks:
            task = tasks.pop(0)
            output = agents[power].run(task)
            conversation.add(
                role=agents[power].agent_name,
                message=output,
            )
    return history_output_formatter(conversation, output_type)


def log_swarm(
    agents: AgentListType,
    tasks: List[str],
    output_type: OutputType = "dict",
) -> Union[Dict[str, Any], List[str]]:
    """
    Implements a logarithmic swarm where agents are selected based on logarithmic progression.

    Args:
        agents (AgentListType): A list of Agent objects to participate in the swarm.
        tasks (List[str]): A list of tasks to be processed by the agents.
        output_type (OutputType, optional): The format of the output. Defaults to "dict".

    Returns:
        Union[Dict[str, Any], List[str]]: The formatted output of the swarm's processing.
            If output_type is "dict", returns a dictionary containing the conversation history.
            If output_type is "list", returns a list of responses.
    """
    conversation = Conversation()
    conversation.add(
        role="User",
        message=tasks,
    )
    for i in range(len(agents)):
        if 2**i < len(agents) and tasks:
            task = tasks.pop(0)
            output = agents[2**i].run(task)
            conversation.add(
                role=agents[2**i].agent_name,
                message=output,
            )
    return history_output_formatter(conversation, output_type)


def exponential_swarm(
    agents: AgentListType,
    tasks: List[str],
    output_type: OutputType = "dict",
) -> Union[Dict[str, Any], List[str]]:
    """
    Implements an exponential swarm where agents are selected based on exponential progression.

    Args:
        agents (AgentListType): A list of Agent objects to participate in the swarm.
        tasks (List[str]): A list of tasks to be processed by the agents.
        output_type (OutputType, optional): The format of the output. Defaults to "dict".

    Returns:
        Union[Dict[str, Any], List[str]]: The formatted output of the swarm's processing.
            If output_type is "dict", returns a dictionary containing the conversation history.
            If output_type is "list", returns a list of responses.
    """
    conversation = Conversation()
    conversation.add(
        role="User",
        message=tasks,
    )

    for i in range(len(agents)):
        index = min(int(2**i), len(agents) - 1)
        if tasks:
            task = tasks.pop(0)
        output = agents[index].run(task)

        conversation.add(
            role=agents[index].agent_name,
            message=output,
        )

    return history_output_formatter(conversation, output_type)


def geometric_swarm(
    agents: AgentListType,
    tasks: List[str],
    output_type: OutputType = "dict",
) -> Union[Dict[str, Any], List[str]]:
    """
    Implements a geometric swarm where agents are selected based on geometric progression.
    Each agent processes tasks in a pattern that follows a geometric sequence.

    Args:
        agents (AgentListType): A list of Agent objects to participate in the swarm.
        tasks (List[str]): A list of tasks to be processed by the agents.
        output_type (OutputType, optional): The format of the output. Defaults to "dict".

    Returns:
        Union[Dict[str, Any], List[str]]: The formatted output of the swarm's processing.
            If output_type is "dict", returns a dictionary containing the conversation history.
            If output_type is "list", returns a list of responses.
    """
    ratio = 2
    conversation = Conversation()
    conversation.add(
        role="User",
        message=tasks,
    )

    for i in range(len(agents)):
        index = min(int(ratio**2), len(agents) - 1)
        if tasks:
            task = tasks.pop(0)
            response = agents[index].run(task)
            conversation.add(
                role=agents[index].agent_name,
                message=response,
            )

    return history_output_formatter(conversation, output_type)


def harmonic_swarm(
    agents: AgentListType,
    tasks: List[str],
    output_type: OutputType = "dict",
) -> Union[Dict[str, Any], List[str]]:
    """
    Implements a harmonic swarm where agents are selected based on harmonic progression.

    Args:
        agents (AgentListType): A list of Agent objects to participate in the swarm.
        tasks (List[str]): A list of tasks to be processed by the agents.
        output_type (OutputType, optional): The format of the output. Defaults to "dict".

    Returns:
        Union[Dict[str, Any], List[str]]: The formatted output of the swarm's processing.
            If output_type is "dict", returns a dictionary containing the conversation history.
            If output_type is "list", returns a list of responses.
    """
    conversation = Conversation()
    conversation.add(
        role="User",
        message=tasks,
    )

    for i in range(1, len(agents) + 1):
        index = min(int(len(agents) / i), len(agents) - 1)
        if tasks:
            task = tasks.pop(0)
            response = agents[index].run(task)
            conversation.add(
                role=agents[index].agent_name,
                message=response,
            )

    return history_output_formatter(conversation, output_type)


def staircase_swarm(
    agents: AgentListType,
    tasks: List[str],
    output_type: OutputType = "dict",
) -> Union[Dict[str, Any], List[str]]:
    """
    Implements a staircase swarm where agents are selected in a step-like pattern.

    Args:
        agents (AgentListType): A list of Agent objects to participate in the swarm.
        tasks (List[str]): A list of tasks to be processed by the agents.
        output_type (OutputType, optional): The format of the output. Defaults to "dict".

    Returns:
        Union[Dict[str, Any], List[str]]: The formatted output of the swarm's processing.
            If output_type is "dict", returns a dictionary containing the conversation history.
            If output_type is "list", returns a list of responses.
    """
    conversation = Conversation()
    conversation.add(
        role="User",
        message=tasks,
    )

    step = len(agents) // 5
    for i in range(len(agents)):
        index = (i // step) * step
        if tasks:
            task = tasks.pop(0)
            response = agents[index].run(task)
            conversation.add(
                role=agents[index].agent_name,
                message=response,
            )

    return history_output_formatter(conversation, output_type)


def sigmoid_swarm(
    agents: AgentListType,
    tasks: List[str],
    output_type: OutputType = "dict",
) -> Union[Dict[str, Any], List[str]]:
    """
    Implements a sigmoid swarm where agents are selected based on sigmoid function.

    Args:
        agents (AgentListType): A list of Agent objects to participate in the swarm.
        tasks (List[str]): A list of tasks to be processed by the agents.
        output_type (OutputType, optional): The format of the output. Defaults to "dict".

    Returns:
        Union[Dict[str, Any], List[str]]: The formatted output of the swarm's processing.
            If output_type is "dict", returns a dictionary containing the conversation history.
            If output_type is "list", returns a list of responses.
    """
    conversation = Conversation()
    conversation.add(
        role="User",
        message=tasks,
    )

    for i in range(len(agents)):
        index = int(len(agents) / (1 + math.exp(-i)))
        if tasks:
            task = tasks.pop(0)
            response = agents[index].run(task)
            conversation.add(
                role=agents[index].agent_name,
                message=response,
            )

    return history_output_formatter(conversation, output_type)


def sinusoidal_swarm(
    agents: AgentListType,
    tasks: List[str],
    output_type: OutputType = "dict",
) -> Union[Dict[str, Any], List[str]]:
    """
    Implements a sinusoidal swarm where agents are selected based on sine function.

    Args:
        agents (AgentListType): A list of Agent objects to participate in the swarm.
        tasks (List[str]): A list of tasks to be processed by the agents.
        output_type (OutputType, optional): The format of the output. Defaults to "dict".

    Returns:
        Union[Dict[str, Any], List[str]]: The formatted output of the swarm's processing.
            If output_type is "dict", returns a dictionary containing the conversation history.
            If output_type is "list", returns a list of responses.
    """
    conversation = Conversation()
    conversation.add(
        role="User",
        message=tasks,
    )

    for i in range(len(agents)):
        index = int((math.sin(i) + 1) / 2 * len(agents))
        if tasks:
            task = tasks.pop(0)
            response = agents[index].run(task)
            conversation.add(
                role=agents[index].agent_name,
                message=response,
            )

    return history_output_formatter(conversation, output_type)


# One-to-One Communication between two agents
def one_to_one(
    sender: Agent,
    receiver: Agent,
    task: str,
    max_loops: int = 1,
    output_type: OutputType = "dict",
) -> Union[Dict[str, Any], List[str]]:
    """
    Implements one-to-one communication between two agents.

    Args:
        sender (Agent): The agent sending the message.
        receiver (Agent): The agent receiving the message.
        task (str): The task to be processed.
        max_loops (int, optional): Maximum number of communication loops. Defaults to 1.
        output_type (OutputType, optional): The format of the output. Defaults to "dict".

    Returns:
        Union[Dict[str, Any], List[str]]: The formatted output of the communication.
            If output_type is "dict", returns a dictionary containing the conversation history.
            If output_type is "list", returns a list of responses.

    Raises:
        ValueError: If sender, receiver, or task is empty.
    """
    conversation = Conversation()
    conversation.add(
        role="User",
        message=task,
    )

    try:
        for _ in range(max_loops):
            # Sender processes the task
            sender_response = sender.run(task)
            conversation.add(
                role=sender.agent_name,
                message=sender_response,
            )

            # Receiver processes the result of the sender
            receiver_response = receiver.run(sender_response)
            conversation.add(
                role=receiver.agent_name,
                message=receiver_response,
            )

        return history_output_formatter(conversation, output_type)

    except Exception as error:
        logger.error(
            f"Error during one_to_one communication: {error}"
        )
        raise error


async def broadcast(
    sender: Agent,
    agents: AgentListType,
    task: str,
    output_type: OutputType = "dict",
) -> Union[Dict[str, Any], List[str]]:
    """
    Implements a broadcast communication pattern where one agent sends to many.

    Args:
        sender (Agent): The agent broadcasting the message.
        agents (AgentListType): List of agents receiving the broadcast.
        task (str): The task to be broadcast.
        output_type (OutputType, optional): The format of the output. Defaults to "dict".

    Returns:
        Union[Dict[str, Any], List[str]]: The formatted output of the broadcast.
            If output_type is "dict", returns a dictionary containing the conversation history.
            If output_type is "list", returns a list of responses.

    Raises:
        ValueError: If sender, agents, or task is empty.
    """
    conversation = Conversation()
    conversation.add(
        role="User",
        message=task,
    )

    if not sender or not agents or not task:
        raise ValueError("Sender, agents, and task cannot be empty.")

    try:
        # First get the sender's broadcast message
        broadcast_message = sender.run(conversation.get_str())

        conversation.add(
            role=sender.agent_name,
            message=broadcast_message,
        )

        # Then have all agents process it
        for agent in agents:
            response = agent.run(conversation.get_str())
            conversation.add(
                role=agent.agent_name,
                message=response,
            )

        return history_output_formatter(conversation, output_type)

    except Exception as error:
        logger.error(f"Error during broadcast: {error}")
        raise error


async def one_to_three(
    sender: Agent,
    agents: AgentListType,
    task: str,
    output_type: OutputType = "dict",
) -> Union[Dict[str, Any], List[str]]:
    """
    Implements one-to-three communication pattern where one agent sends to three others.

    Args:
        sender (Agent): The agent sending the message.
        agents (AgentListType): List of three agents receiving the message.
        task (str): The task to be processed.
        output_type (OutputType, optional): The format of the output. Defaults to "dict".

    Returns:
        Union[Dict[str, Any], List[str]]: The formatted output of the communication.
            If output_type is "dict", returns a dictionary containing the conversation history.
            If output_type is "list", returns a list of responses.

    Raises:
        ValueError: If sender, agents, or task is empty, or if agents list doesn't contain exactly 3 agents.
    """
    if len(agents) != 3:
        raise ValueError("The number of agents must be exactly 3.")

    if not task or not sender:
        raise ValueError("Sender and task cannot be empty.")

    conversation = Conversation()

    conversation.add(
        role="User",
        message=task,
    )

    try:
        # Get sender's message
        sender_message = sender.run(conversation.get_str())
        conversation.add(
            role=sender.agent_name,
            message=sender_message,
        )

        # Have each receiver process the message
        for agent in agents:
            response = agent.run(conversation.get_str())
            conversation.add(
                role=agent.agent_name,
                message=response,
            )

        return history_output_formatter(conversation, output_type)

    except Exception as error:
        logger.error(f"Error in one_to_three: {error}")
        raise error
