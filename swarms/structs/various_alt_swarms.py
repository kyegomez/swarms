from typing import Dict, List, Union

from loguru import logger

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.omni_agent_types import AgentListType
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)


class OneToOne:
    """
    Facilitates one-to-one communication between two agents.
    """

    def __init__(
        self,
        sender: Agent,
        receiver: Agent,
        name: str = "OneToOne",
        description: str = "A one-to-one communication pattern between two agents",
        output_type: str = "dict",
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
        self.conversation = Conversation()

    def run(
        self, task: str, max_loops: int = 1
    ) -> Union[Dict, List, str]:
        """
        Run the one-to-one communication with the given task

        Args:
            task: Task to be processed
            max_loops: Number of exchange iterations

        Returns:
            Union[Dict, List, str]: The conversation history in the requested format
        """
        if not self.sender or not self.receiver or not task:
            raise ValueError(
                "Sender, receiver, and task cannot be empty."
            )

        responses = []

        try:
            for loop in range(max_loops):
                # Sender processes the task
                sender_response = self.sender.run(task)
                self.conversation.add(
                    role=self.sender.agent_name,
                    content=sender_response,
                )
                responses.append(sender_response)

                # Receiver processes the result of the sender
                receiver_response = self.receiver.run(sender_response)
                self.conversation.add(
                    role=self.receiver.agent_name,
                    content=receiver_response,
                )
                responses.append(receiver_response)

                # Update task for next loop if needed
                if loop < max_loops - 1:
                    task = receiver_response

        except Exception as error:
            logger.error(
                f"Error during one_to_one communication: {error}"
            )
            raise error

        return history_output_formatter(
            self.conversation, self.output_type
        )


class Broadcast:
    """
    Facilitates broadcasting from one agent to many agents.
    """

    def __init__(
        self,
        sender: Agent,
        receivers: AgentListType,
        name: str = "Broadcast",
        description: str = "A broadcast communication pattern from one agent to many agents",
        output_type: str = "dict",
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
        self.receivers = (
            [agent for sublist in receivers for agent in sublist]
            if isinstance(receivers[0], list)
            else receivers
        )
        self.name = name
        self.description = description
        self.output_type = output_type
        self.conversation = Conversation()

    def run(self, task: str) -> Union[Dict, List, str]:
        """
        Run the broadcast communication with the given task

        Args:
            task: Task to be processed

        Returns:
            Union[Dict, List, str]: The conversation history in the requested format
        """
        if not self.sender or not self.receivers or not task:
            raise ValueError(
                "Sender, receivers, and task cannot be empty."
            )

        try:
            # First get the sender's broadcast message
            broadcast_message = self.sender.run(task)
            self.conversation.add(
                role=self.sender.agent_name,
                content=broadcast_message,
            )

            # Then have all receivers process it
            for agent in self.receivers:
                response = agent.run(broadcast_message)
                self.conversation.add(
                    role=agent.agent_name,
                    content=response,
                )

            return history_output_formatter(
                self.conversation, self.output_type
            )

        except Exception as error:
            logger.error(f"Error during broadcast: {error}")
            raise error


class OneToThree:
    """
    Facilitates one-to-three communication from one agent to exactly three agents.
    """

    def __init__(
        self,
        sender: Agent,
        receivers: AgentListType,
        name: str = "OneToThree",
        description: str = "A one-to-three communication pattern from one agent to exactly three agents",
        output_type: str = "dict",
    ):
        """
        Initialize the OneToThree communication.

        Args:
            sender: The sender agent
            receivers: List of exactly three receiver agents
            name: Name of the communication pattern
            description: Description of the communication pattern's purpose
            output_type: Type of output format, one of 'dict', 'list', 'string', 'json', 'yaml', 'xml', etc.
        """
        if len(receivers) != 3:
            raise ValueError(
                "The number of receivers must be exactly 3."
            )

        self.sender = sender
        self.receivers = receivers
        self.name = name
        self.description = description
        self.output_type = output_type
        self.conversation = Conversation()

    def run(self, task: str) -> Union[Dict, List, str]:
        """
        Run the one-to-three communication with the given task

        Args:
            task: Task to be processed

        Returns:
            Union[Dict, List, str]: The conversation history in the requested format
        """
        if not self.sender or not task:
            raise ValueError("Sender and task cannot be empty.")

        try:
            # Get sender's message
            sender_message = self.sender.run(task)
            self.conversation.add(
                role=self.sender.agent_name,
                content=sender_message,
            )

            # Have each receiver process the message
            for i, agent in enumerate(self.receivers):
                response = agent.run(sender_message)
                self.conversation.add(
                    role=agent.agent_name,
                    content=response,
                )

            return history_output_formatter(
                self.conversation, self.output_type
            )

        except Exception as error:
            logger.error(f"Error in OneToThree: {error}")
            raise error
