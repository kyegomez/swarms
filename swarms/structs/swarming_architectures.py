import asyncio
import math
from typing import List

from swarms.structs.agent import Agent
from swarms.utils.logger import logger


def circular_swarm(agents: List[Agent], tasks: List[str]):
    ball = 0
    while tasks:
        task = tasks.pop(0)
        agents[ball].run(task)
        ball = (ball + 1) % len(agents)


def linear_swarm(agents: List[Agent], tasks: List[str]):
    for i in range(len(agents)):
        if tasks:
            task = tasks.pop(0)
            agents[i].run(task)


def star_swarm(agents: List[Agent], tasks: List[str]):
    center_agent = agents[0]
    for task in tasks:
        center_agent.run(task)
        for agent in agents[1:]:
            agent.run(task)


def mesh_swarm(agents: List[Agent], tasks: List[str]):
    task_queue = tasks.copy()
    while task_queue:
        for agent in agents:
            if task_queue:
                task = task_queue.pop(0)
                agent.run(task)


def grid_swarm(agents: List[Agent], tasks: List[str]):
    grid_size = int(
        len(agents) ** 0.5
    )  # Assuming agents can form a perfect square grid
    for i in range(grid_size):
        for j in range(grid_size):
            if tasks:
                task = tasks.pop(0)
                agents[i * grid_size + j].run(task)


def pyramid_swarm(agents: List[Agent], tasks: List[str]):
    levels = int(
        (-1 + (1 + 8 * len(agents)) ** 0.5) / 2
    )  # Assuming agents can form a perfect pyramid
    for i in range(levels):
        for j in range(i + 1):
            if tasks:
                task = tasks.pop(0)
                agents[int(i * (i + 1) / 2 + j)].run(task)


def fibonacci_swarm(agents: List[Agent], tasks: List[str]):
    fib = [1, 1]
    while len(fib) < len(agents):
        fib.append(fib[-1] + fib[-2])
    for i in range(len(fib)):
        for j in range(fib[i]):
            if tasks:
                task = tasks.pop(0)
                agents[int(sum(fib[:i]) + j)].run(task)


def prime_swarm(agents: List[Agent], tasks: List[str]):
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


def log_swarm(agents: List[Agent], tasks: List[str]):
    for i in range(len(agents)):
        if 2**i < len(agents) and tasks:
            task = tasks.pop(0)
            agents[2**i].run(task)


def exponential_swarm(agents: List[Agent], tasks: List[str]):
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


def harmonic_swarm(agents: List[Agent], tasks: List[str]):
    for i in range(1, len(agents) + 1):
        index = min(int(len(agents) / i), len(agents) - 1)
        if tasks:
            task = tasks.pop(0)
            agents[index].run(task)


def staircase_swarm(agents: List[Agent], task: str):
    step = len(agents) // 5
    for i in range(len(agents)):
        index = (i // step) * step
        agents[index].run(task)


def sigmoid_swarm(agents: List[Agent], task: str):
    for i in range(len(agents)):
        index = int(len(agents) / (1 + math.exp(-i)))
        agents[index].run(task)


def sinusoidal_swarm(agents: List[Agent], task: str):
    for i in range(len(agents)):
        index = int((math.sin(i) + 1) / 2 * len(agents))
        agents[index].run(task)


async def one_to_three(sender: Agent, agents: List[Agent], task: str):
    """
    Sends a message from the sender agent to three other agents.

    Args:
        sender (Agent): The agent sending the message.
        agents (List[Agent]): The list of agents to receive the message.
        task (str): The message to be sent.

    Raises:
        Exception: If there is an error while sending the message.

    Returns:
        None
    """
    try:
        receive_tasks = []
        for agent in agents:
            receive_tasks.append(
                agent.receive_message(sender.ai_name, task)
            )

        await asyncio.gather(*receive_tasks)
    except Exception as error:
        logger.error(
            f"[ERROR][CLASS: Agent][METHOD: one_to_three] {error}"
        )
        raise error


async def broadcast(
    sender: Agent,
    agents: List[Agent],
    task: str,
):
    """
    Broadcasts a message from the sender agent to a list of agents.

    Args:
        sender (Agent): The agent sending the message.
        agents (List[Agent]): The list of agents to receive the message.
        task (str): The message to be broadcasted.

    Raises:
        Exception: If an error occurs during the broadcast.

    Returns:
        None
    """
    try:
        receive_tasks = []
        for agent in agents:
            receive_tasks.append(
                agent.receive_message(sender.ai_name, task)
            )

        await asyncio.gather(*receive_tasks)
    except Exception as error:
        logger.error(f"[ERROR][CLASS: Agent][METHOD: broadcast] {error}")
        raise error


async def one_to_one(
    sender: Agent,
    receiver: Agent,
    task: str,
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
        await receiver.receive_message(sender.ai_name, task)
    except Exception as error:
        logger.error(f"[ERROR][CLASS: Agent][METHOD: one_to_one] {error}")
        raise error
