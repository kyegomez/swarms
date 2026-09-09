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

logger = initialize_logger(log_folder="one_to_three")


def _check_receivers(receivers: AgentListType) -> None:
    if receivers is None or len(receivers) != 3:
        raise ValueError("The number of receivers must be exactly 3.")


def one_to_three(
    sender: Agent,
    receivers: AgentListType,
    task: str,
    output_type: OutputType = "dict",
) -> Union[Dict[str, Any], List[str], str]:
    """
    One-to-three communication from one agent to exactly three agents.

    The sender answers the task, then each of the three receivers reads the
    shared conversation and replies to it in turn.

    Args:
        sender (Agent): The agent sending the message.
        receivers (AgentListType): Exactly three receiving agents.
        task (str): The task to be processed.
        output_type (OutputType, optional): The format of the output.
            Defaults to "dict".

    Returns:
        Union[Dict[str, Any], List[str], str]: The conversation history in
            the requested format.

    Raises:
        ValueError: If sender or task is empty, or there are not exactly
            three receivers.
    """
    _check_receivers(receivers)
    if not sender or not task:
        raise ValueError("Sender and task cannot be empty.")

    conversation = Conversation()
    conversation.add(role="User", content=task)

    try:
        run_on_conversation(sender, conversation)

        for agent in receivers:
            run_on_conversation(agent, conversation)
    except Exception as error:
        logger.error(f"Error in one_to_three: {error}")
        raise error

    return history_output_formatter(conversation, output_type)


class OneToThree:
    """
    Facilitates one-to-three communication from one agent to exactly three agents.

    A reusable wrapper around :func:`one_to_three` that holds the sender,
    the three receivers and the output format.
    """

    def __init__(
        self,
        sender: Agent,
        receivers: AgentListType,
        name: str = "OneToThree",
        description: str = "A one-to-three communication pattern from one agent to exactly three agents",
        output_type: OutputType = "dict",
    ):
        """
        Initialize the OneToThree communication.

        Args:
            sender: The sender agent
            receivers: List of exactly three receiver agents
            name: Name of the communication pattern
            description: Description of the communication pattern's purpose
            output_type: Type of output format, one of 'dict', 'list', 'string', 'json', 'yaml', 'xml', etc.

        Raises:
            ValueError: If there are not exactly three receivers.
        """
        _check_receivers(receivers)

        self.sender = sender
        self.receivers = receivers
        self.name = name
        self.description = description
        self.output_type = output_type

    def run(self, task: str) -> Union[Dict[str, Any], List[str], str]:
        """
        Run the one-to-three communication with the given task.

        Args:
            task: Task to be processed

        Returns:
            Union[Dict, List, str]: The conversation history in the requested format
        """
        return one_to_three(
            sender=self.sender,
            receivers=self.receivers,
            task=task,
            output_type=self.output_type,
        )
