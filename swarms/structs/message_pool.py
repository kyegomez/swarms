import hashlib
from time import time_ns
from typing import Callable, List, Optional, Sequence, Union

from swarms.structs.agent import Agent
from swarms.utils.loguru_logger import logger
from swarms.structs.base_swarm import BaseSwarm


def _hash(input: str):
    """
    Hashes the input string using SHA256 algorithm.

    Args:
        input (str): The string to be hashed.

    Returns:
        str: The hexadecimal representation of the hash value.
    """
    hex_dig = hashlib.sha256(input.encode("utf-8")).hexdigest()
    return hex_dig


def msg_hash(
    agent: Agent, content: str, turn: int, msg_type: str = "text"
):
    """
    Generate a hash value for a message.

    Args:
        agent (Agent): The agent sending the message.
        content (str): The content of the message.
        turn (int): The turn number of the message.
        msg_type (str, optional): The type of the message. Defaults to "text".

    Returns:
        int: The hash value of the message.
    """
    time = time_ns()
    return _hash(
        f"agent: {agent.agent_name}\ncontent: {content}\ntimestamp:"
        f" {str(time)}\nturn: {turn}\nmsg_type: {msg_type}"
    )


class MessagePool(BaseSwarm):
    """
    A class representing a message pool for agents in a swarm.

    Attributes:
        agents (Optional[Sequence[Agent]]): The list of agents in the swarm.
        moderator (Optional[Agent]): The moderator agent.
        turns (Optional[int]): The number of turns.
        routing_function (Optional[Callable]): The routing function for message distribution.
        show_names (Optional[bool]): Flag indicating whether to show agent names.
        messages (List[Dict]): The list of messages in the pool.

    Examples:
    >>> from swarms.structs.agent import Agent
    >>> from swarms.structs.message_pool import MessagePool
    >>> agent1 = Agent(agent_name="agent1")
    >>> agent2 = Agent(agent_name="agent2")
    >>> agent3 = Agent(agent_name="agent3")
    >>> moderator = Agent(agent_name="moderator")
    >>> agents = [agent1, agent2, agent3]
    >>> message_pool = MessagePool(agents=agents, moderator=moderator, turns=5)
    >>> message_pool.add(agent=agent1, content="Hello, agent2!", turn=1)
    >>> message_pool.add(agent=agent2, content="Hello, agent1!", turn=1)
    >>> message_pool.add(agent=agent3, content="Hello, agent1!", turn=1)
    >>> message_pool.get_all_messages()
    [{'agent': Agent(agent_name='agent1'), 'content': 'Hello, agent2!', 'turn': 1, 'visible_to': 'all', 'logged': True}, {'agent': Agent(agent_name='agent2'), 'content': 'Hello, agent1!', 'turn': 1, 'visible_to': 'all', 'logged': True}, {'agent': Agent(agent_name='agent3'), 'content': 'Hello, agent1!', 'turn': 1, 'visible_to': 'all', 'logged': True}]
    >>> message_pool.get_visible_messages(agent=agent1, turn=1)
    [{'agent': Agent(agent_name='agent1'), 'content': 'Hello, agent2!', 'turn': 1, 'visible_to': 'all', 'logged': True}, {'agent': Agent(agent_name='agent2'), 'content': 'Hello, agent1!', 'turn': 1, 'visible_to': 'all', 'logged': True}, {'agent': Agent(agent_name='agent3'), 'content': 'Hello, agent1!', 'turn': 1, 'visible_to': 'all', 'logged': True}]
    >>> message_pool.get_visible_messages(agent=agent2, turn=1)
    [{'agent': Agent(agent_name='agent1'), 'content': 'Hello, agent2!', 'turn': 1, 'visible_to': 'all', 'logged': True}, {'agent': Agent(agent_name='agent2'), 'content': 'Hello, agent1!', 'turn': 1, 'visible_to': 'all', 'logged': True}, {'agent': Agent(agent_name='agent3'), 'content': 'Hello, agent1!', 'turn': 1, 'visible_to': 'all', 'logged': True}]
    """

    def __init__(
        self,
        agents: Optional[Sequence[Agent]] = None,
        moderator: Optional[Agent] = None,
        turns: Optional[int] = 5,
        routing_function: Optional[Callable] = None,
        show_names: Optional[bool] = False,
        autosave: Optional[bool] = False,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.agent = agents
        self.moderator = moderator
        self.turns = turns
        self.routing_function = routing_function
        self.show_names = show_names
        self.autosave = autosave

        self.messages = []

        logger.info("MessagePool initialized")
        logger.info(f"Number of agents: {len(agents)}")
        logger.info(
            f"Agents: {[agent.agent_name for agent in agents]}"
        )
        logger.info(f"moderator: {moderator.agent_name} is available")
        logger.info(f"Number of turns: {turns}")

    def add(
        self,
        agent: Agent,
        content: str,
        turn: int,
        visible_to: Union[str, List[str]] = "all",
        logged: bool = True,
    ):
        """
        Add a message to the pool.

        Args:
            agent (Agent): The agent sending the message.
            content (str): The content of the message.
            turn (int): The turn number.
            visible_to (Union[str, List[str]], optional): The agents who can see the message. Defaults to "all".
            logged (bool, optional): Flag indicating whether the message should be logged. Defaults to True.
        """

        self.messages.append(
            {
                "agent": agent,
                "content": content,
                "turn": turn,
                "visible_to": visible_to,
                "logged": logged,
            }
        )
        logger.info(f"Message added: {content}")

    def reset(self):
        """
        Reset the message pool.
        """
        self.messages = []
        logger.info("MessagePool reset")

    def last_turn(self):
        """
        Get the last turn number.

        Returns:
            int: The last turn number.
        """
        if len(self.messages) == 0:
            return 0
        else:
            return self.messages[-1]["turn"]

    @property
    def last_message(self):
        """
        Get the last message in the pool.

        Returns:
            dict: The last message.
        """
        if len(self.messages) == 0:
            return None
        else:
            return self.messages[-1]

    def get_all_messages(self):
        """
        Get all messages in the pool.

        Returns:
            List[Dict]: The list of all messages.
        """
        return self.messages

    def get_visible_messages(self, agent: Agent, turn: int):
        """
        Get the visible messages for a given agent and turn.

        Args:
            agent (Agent): The agent.
            turn (int): The turn number.

        Returns:
            List[Dict]: The list of visible messages.
        """
        # Get the messages before the current turn
        prev_messages = [
            message
            for message in self.messages
            if message["turn"] < turn
        ]

        visible_messages = []
        for message in prev_messages:
            if (
                message["visible_to"] == "all"
                or agent.agent_name in message["visible_to"]
            ):
                visible_messages.append(message)
        return visible_messages

    # def query(self, query: str):
    #     """
    #     Query a message from the messages list and then pass it to the moderator
    #     """
    #     return [
    #         (mod, content)
    #         for mod, content, _ in self.messages  # Add an underscore to ignore the rest of the elements
    #         if query in content
    #     ]
