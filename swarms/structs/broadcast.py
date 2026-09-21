from typing import Any, Dict, List, Union

from swarms.structs.agent import Agent
from swarms.structs.context_utils import run_on_conversation
from swarms.structs.conversation import Conversation
from swarms.structs.omni_agent_types import AgentListType
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType

logger = initialize_logger(log_folder="broadcast")


def _flatten(receivers: AgentListType) -> List[Agent]:
    """Accept either a flat list of agents or a list of lists."""
    if receivers and isinstance(receivers[0], list):
        return [agent for group in receivers for agent in group]
    return list(receivers)


def _broadcast(
    sender: Agent,
    receivers: AgentListType,
    task: str,
    output_type: OutputType = "dict",
) -> Union[Dict[str, Any], List[str], str]:
    if not sender or not receivers or not task:
        raise ValueError(
            "Sender, receivers, and task cannot be empty."
        )

    conversation = Conversation()
    conversation.add(role="User", content=task)

    try:
        run_on_conversation(sender, conversation)

        for agent in _flatten(receivers):
            run_on_conversation(agent, conversation)
    except Exception as error:
        logger.error(f"Error during broadcast: {error}")
        raise error

    return history_output_formatter(conversation, output_type)


async def broadcast(
    sender: Agent,
    agents: AgentListType,
    task: str,
    output_type: OutputType = "dict",
) -> Union[Dict[str, Any], List[str], str]:
    """
    Broadcast communication from one agent to many.

    The sender answers the task, then every receiver reads the shared
    conversation and replies to it in turn.

    Args:
        sender (Agent): The agent broadcasting the message.
        agents (AgentListType): The agents receiving the broadcast, as a
            flat list or a list of lists.
        task (str): The task to be broadcast.
        output_type (OutputType, optional): The format of the output.
            Defaults to "dict".

    Returns:
        Union[Dict[str, Any], List[str], str]: The conversation history in
            the requested format.

    Raises:
        ValueError: If sender, agents, or task is empty.
    """
    return _broadcast(sender, agents, task, output_type)


class Broadcast:
    """
    Facilitates broadcasting from one agent to many agents.

    A reusable wrapper around :func:`broadcast` that holds the sender, the
    receivers and the output format, so the same group can be run on many
    tasks. ``run`` is synchronous.
    """

    def __init__(
        self,
        sender: Agent,
        receivers: AgentListType,
        name: str = "Broadcast",
        description: str = "A broadcast communication pattern from one agent to many agents",
        output_type: OutputType = "dict",
    ):
        """
        Initialize the Broadcast communication.

        Args:
            sender: The sender agent
            receivers: List of receiver agents
            name: Name of the communication pattern
            description: Description of the communication pattern's purpose
            output_type: Type of output format, one of 'dict', 'list', 'string', 'json', 'yaml', 'xml', etc.
        """
        self.sender = sender
        self.receivers = _flatten(receivers)
        self.name = name
        self.description = description
        self.output_type = output_type

    def run(self, task: str) -> Union[Dict[str, Any], List[str], str]:
        """
        Run the broadcast communication with the given task.

        Args:
            task: Task to be processed

        Returns:
            Union[Dict, List, str]: The conversation history in the requested format
        """
        return _broadcast(
            sender=self.sender,
            receivers=self.receivers,
            task=task,
            output_type=self.output_type,
        )
