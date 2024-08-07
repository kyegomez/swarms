import asyncio
import math
from typing import List

from swarms.structs.agent import Agent
from swarms.utils.loguru_logger import logger
from swarms.structs.conversation import Conversation
from swarms.structs.concat import concat_strings
from swarms.structs.omni_agent_types import AgentListType

# from swarms.structs.swarm_registry import swarm_registry, SwarmRegistry


# @swarm_registry
def circular_swarm(
    name: str = "Circular Swarm",
    description: str = "A circular swarm is a type of swarm where agents pass tasks in a circular manner.",
    goal: str = None,
    agents: AgentListType = None,
    tasks: List[str] = None,
    return_full_history: bool = True,
):
    if not agents:
        raise ValueError("Agents list cannot be empty.")

    if not tasks:
        raise ValueError("Tasks list cannot be empty.")

    conversation = Conversation(
        time_enabled=True,
    )

    responses = []

    for task in tasks:
        for agent in agents:
            # Log the task
            out = agent.run(task)
            # print(f"Task: {task}, Response {out}")
            # prompt = f"Task: {task}, Response {out}"
            logger.info(f"Agent: {agent.agent_name} Response {out}")

            conversation.add(
                role=agent.agent_name,
                content=out,
            )

            # Response list
            responses.append(out)

    if return_full_history:
        return conversation.return_history_as_string()
    else:
        return responses


# @swarm_registry()
def linear_swarm(
    name: str = "Linear Swarm",
    description: str = "A linear swarm is a type of swarm where agents pass tasks in a linear manner.",
    agents: AgentListType = None,
    tasks: List[str] = None,
    conversation: Conversation = None,
    return_full_history: bool = True,
):
    if not agents:
        raise ValueError("Agents list cannot be empty.")

    if not tasks:
        raise ValueError("Tasks list cannot be empty.")

    if not conversation:
        conversation = Conversation(
            time_enabled=True,
        )

    responses = []

    for i in range(len(agents)):
        if tasks:
            task = tasks.pop(0)
            out = agents[i].run(task)

            conversation.add(
                role=agents[i].agent_name,
                content=f"Task: {task}, Response {out}",
            )

            responses.append(out)

    if return_full_history:
        return conversation.return_history_as_string()
    else:
        return responses


# print(SwarmRegistry().list_swarms())

# def linear_swarm(agents: AgentListType, tasks: List[str]):
#     logger.info(f"Running linear swarm with {len(agents)} agents")
#     for i in range(len(agents)):
#         if tasks:
#             task = tasks.pop(0)
#             agents[i].run(task)


def star_swarm(agents: AgentListType, tasks: List[str]) -> str:
    logger.info(
        f"Running star swarm with {len(agents)} agents and {len(tasks)} tasks"
    )

    if not agents:
        raise ValueError("Agents list cannot be empty.")

    if not tasks:
        raise ValueError("Tasks list cannot be empty.")

    conversation = Conversation(time_enabled=True)
    center_agent = agents[0]

    responses = []

    for task in tasks:

        out = center_agent.run(task)
        log = f"Agent: {center_agent.agent_name} Response {out}"
        logger.info(log)
        conversation.add(center_agent.agent_name, out)
        responses.append(out)

        for agent in agents[1:]:

            output = agent.run(task)
            log_two = f"Agent: {agent.agent_name} Response {output}"
            logger.info(log_two)
            conversation.add(agent.agent_name, output)
            responses.append(out)

    out = concat_strings(responses)
    print(out)

    return out


def mesh_swarm(agents: AgentListType, tasks: List[str]):
    task_queue = tasks.copy()
    while task_queue:
        for agent in agents:
            if task_queue:
                task = task_queue.pop(0)
                agent.run(task)


def grid_swarm(agents: AgentListType, tasks: List[str]):
    grid_size = int(
        len(agents) ** 0.5
    )  # Assuming agents can form a perfect square grid
    for i in range(grid_size):
        for j in range(grid_size):
            if tasks:
                task = tasks.pop(0)
                agents[i * grid_size + j].run(task)


def pyramid_swarm(agents: AgentListType, tasks: List[str]):
    levels = int(
        (-1 + (1 + 8 * len(agents)) ** 0.5) / 2
    )  # Assuming agents can form a perfect pyramid
    for i in range(levels):
        for j in range(i + 1):
            if tasks:
                task = tasks.pop(0)
                agents[int(i * (i + 1) / 2 + j)].run(task)


def fibonacci_swarm(agents: AgentListType, tasks: List[str]):
    fib = [1, 1]
    while len(fib) < len(agents):
        fib.append(fib[-1] + fib[-2])
    for i in range(len(fib)):
        for j in range(fib[i]):
            if tasks:
                task = tasks.pop(0)
                agents[int(sum(fib[:i]) + j)].run(task)


def prime_swarm(agents: AgentListType, tasks: List[str]):
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
            agents[prime].run(task)


def power_swarm(agents: List[str], tasks: List[str]):
    powers = [2**i for i in range(int(len(agents) ** 0.5))]
    for power in powers:
        if power < len(agents) and tasks:
            task = tasks.pop(0)
            agents[power].run(task)


def log_swarm(agents: AgentListType, tasks: List[str]):
    for i in range(len(agents)):
        if 2**i < len(agents) and tasks:
            task = tasks.pop(0)
            agents[2**i].run(task)


def exponential_swarm(agents: AgentListType, tasks: List[str]):
    for i in range(len(agents)):
        index = min(int(2**i), len(agents) - 1)
        if tasks:
            task = tasks.pop(0)
            agents[index].run(task)


def geometric_swarm(agents, tasks):
    ratio = 2
    for i in range(range(len(agents))):
        index = min(int(ratio**2), len(agents) - 1)
        if tasks:
            task = tasks.pop(0)
            agents[index].run(task)


def harmonic_swarm(agents: AgentListType, tasks: List[str]):
    for i in range(1, len(agents) + 1):
        index = min(int(len(agents) / i), len(agents) - 1)
        if tasks:
            task = tasks.pop(0)
            agents[index].run(task)


def staircase_swarm(agents: AgentListType, task: str):
    step = len(agents) // 5
    for i in range(len(agents)):
        index = (i // step) * step
        agents[index].run(task)


def sigmoid_swarm(agents: AgentListType, task: str):
    for i in range(len(agents)):
        index = int(len(agents) / (1 + math.exp(-i)))
        agents[index].run(task)


def sinusoidal_swarm(agents: AgentListType, task: str):
    for i in range(len(agents)):
        index = int((math.sin(i) + 1) / 2 * len(agents))
        agents[index].run(task)


async def one_to_three(sender: Agent, agents: AgentListType, task: str):
    """
    Sends a message from the sender agent to three other agents.

    Args:
        sender (Agent): The agent sending the message.
        agents (AgentListType): The list of agents to receive the message.
        task (str): The message to be sent.

    Raises:
        Exception: If there is an error while sending the message.

    Returns:
        None
    """
    if len(agents) != 3:
        raise ValueError("The number of agents must be exactly 3.")

    if not task:
        raise ValueError("The task cannot be empty.")

    if not sender:
        raise ValueError("The sender cannot be empty.")

    try:
        receive_tasks = []
        for agent in agents:
            receive_tasks.append(
                agent.receive_message(sender.agent_name, task)
            )

        await asyncio.gather(*receive_tasks)
    except Exception as error:
        logger.error(
            f"[ERROR][CLASS: Agent][METHOD: one_to_three] {error}"
        )
        raise error


async def broadcast(
    sender: Agent,
    agents: AgentListType,
    task: str,
):
    """
    Broadcasts a message from the sender agent to a list of agents.

    Args:
        sender (Agent): The agent sending the message.
        agents (AgentListType): The list of agents to receive the message.
        task (str): The message to be broadcasted.

    Raises:
        Exception: If an error occurs during the broadcast.

    Returns:
        None
    """
    if not sender:
        raise ValueError("The sender cannot be empty.")

    if not agents:
        raise ValueError("The agents list cannot be empty.")

    if not task:
        raise ValueError("The task cannot be empty.")

    try:
        receive_tasks = []
        for agent in agents:
            receive_tasks.append(
                agent.receive_message(sender.agent_name, task)
            )

        await asyncio.gather(*receive_tasks)
    except Exception as error:
        logger.error(f"[ERROR][CLASS: Agent][METHOD: broadcast] {error}")
        raise error


def one_to_one(
    sender: Agent,
    receiver: Agent,
    task: str,
    max_loops: int = 1,
):
    """
    Sends a message from the sender agent to the receiver agent.

    Args:
        sender (Agent): The agent sending the message.
        receiver (Agent): The agent to receive the message.
        task (str): The message to be sent.

    Raises:
        Exception: If an error occurs during the message sending.

    Returns:
        None
    """
    try:
        responses = []
        responses.append(task)
        for i in range(max_loops):

            # Run the agent on the task then pass the response to the receiver
            response = sender.run(task)
            log = f"Agent {sender.agent_name} Response: {response}"
            responses.append(log)

            # Send the response to the receiver
            out = receiver.run(concat_strings(responses))
            responses.append(out)

        return concat_strings(responses)
    except Exception as error:
        logger.error(f"[ERROR][CLASS: Agent][METHOD: one_to_one] {error}")
        raise error
