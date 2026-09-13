from typing import Any, Dict, List, Union

from swarms.structs.agent import Agent
from swarms.structs.context_utils import run_on_conversation
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType

logger = initialize_logger(log_folder="one_to_one")


def one_to_one(
    sender: Agent,
    receiver: Agent,
    task: str,
    max_loops: int = 1,
    output_type: OutputType = "dict",
) -> Union[Dict[str, Any], List[str], str]:
    """
    One-to-one communication between two agents.

    The sender answers the task, the receiver answers the sender, and the
    exchange repeats ``max_loops`` times. Each agent reads the shared
    conversation as typed turns, so it can tell its own earlier messages
    from the other agent's.

    Args:
        sender (Agent): The agent that speaks first.
        receiver (Agent): The agent that replies.
        task (str): The task to be processed.
        max_loops (int, optional): Number of sender/receiver exchanges.
            Defaults to 1.
        output_type (OutputType, optional): The format of the output.
            Defaults to "dict".

    Returns:
        Union[Dict[str, Any], List[str], str]: The conversation history in
            the requested format.

    Raises:
        ValueError: If sender, receiver, or task is empty.
    """
    if not sender or not receiver or not task:
        raise ValueError(
            "Sender, receiver, and task cannot be empty."
        )

    conversation = Conversation()
    conversation.add(role="User", content=task)

    try:
        for _ in range(max_loops):
            run_on_conversation(sender, conversation)
            run_on_conversation(receiver, conversation)
    except Exception as error:
        logger.error(
            f"Error during one_to_one communication: {error}"
        )
        raise error

    return history_output_formatter(conversation, output_type)


class OneToOne:
    """
    Facilitates one-to-one communication between two agents.

    A reusable wrapper around :func:`one_to_one` that holds the two agents
    and the output format, so the same pair can be run on many tasks.
    """

    def __init__(
        self,
        sender: Agent,
        receiver: Agent,
        name: str = "OneToOne",
        description: str = "A one-to-one communication pattern between two agents",
        output_type: OutputType = "dict",
    ):
        """
        Initialize the OneToOne communication.

        Args:
            sender: The sender agent
            receiver: The receiver agent
            name: Name of the communication pattern
            description: Description of the communication pattern's purpose
            output_type: Type of output format, one of 'dict', 'list', 'string', 'json', 'yaml', 'xml', etc.
        """
        self.sender = sender
        self.receiver = receiver
        self.name = name
        self.description = description
        self.output_type = output_type

    def run(
        self, task: str, max_loops: int = 1
    ) -> Union[Dict[str, Any], List[str], str]:
        """
        Run the one-to-one communication with the given task.

        Args:
            task: Task to be processed
            max_loops: Number of exchange iterations

        Returns:
            Union[Dict, List, str]: The conversation history in the requested format
        """
        return one_to_one(
            sender=self.sender,
            receiver=self.receiver,
            task=task,
            max_loops=max_loops,
            output_type=self.output_type,
        )
