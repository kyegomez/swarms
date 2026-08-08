from typing import Dict, List, Union

from loguru import logger

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.omni_agent_types import AgentListType
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)


# Base Swarm class that all other swarm types will inherit from
class BaseSwarm:
    def __init__(
        self,
        agents: AgentListType,
        name: str = "BaseSwarm",
        description: str = "A base swarm implementation",
        output_type: str = "dict",
    ):
        """
        Initialize the BaseSwarm with agents, name, description, and output type.

        Args:
            agents: List of Agent objects or nested list of Agent objects
            name: Name of the swarm
            description: Description of the swarm's purpose
            output_type: Type of output format, one of 'dict', 'list', 'string', 'json', 'yaml', 'xml', etc.
        """
        # Ensure agents is a flat list of Agent objects
        self.agents = (
            [agent for sublist in agents for agent in sublist]
            if isinstance(agents[0], list)
            else agents
        )
        self.name = name
        self.description = description
        self.output_type = output_type
        self.conversation = Conversation()

    def run(self, tasks: List[str]) -> Union[Dict, List, str]:
        """
        Run the swarm with the given tasks

        Args:
            tasks: List of tasks to be processed

        Returns:
            Union[Dict, List, str]: The conversation history in the requested format
        """
        if not self.agents or not tasks:
            raise ValueError(
                "Agents and tasks lists cannot be empty."
            )

        # Implementation will be overridden by child classes
        raise NotImplementedError(
            "This method should be implemented by child classes"
        )

    def _format_return(self) -> Union[Dict, List, str]:
        """Format the return value based on the output_type using history_output_formatter"""
        return history_output_formatter(
            self.conversation, self.output_type
        )


class CircularSwarm(BaseSwarm):
    """
    Implements a circular swarm where agents pass tasks in a circular manner.
    """

    def __init__(
        self,
        agents: AgentListType,
        name: str = "CircularSwarm",
        description: str = "A circular swarm where agents pass tasks in a circular manner",
        output_type: str = "dict",
    ):
        """
        Initialize the CircularSwarm.

        Args:
            agents: List of Agent objects or nested list of Agent objects
            name: Name of the swarm
            description: Description of the swarm's purpose
            output_type: Type of output format, one of 'dict', 'list', 'string', 'json', 'yaml', 'xml', etc.
        """
        super().__init__(agents, name, description, output_type)

    def run(self, tasks: List[str]) -> Union[Dict, List, str]:
        """
        Run the circular swarm with the given tasks

        Args:
            tasks: List of tasks to be processed

        Returns:
            Union[Dict, List, str]: The conversation history in the requested format
        """
        if not self.agents or not tasks:
            raise ValueError(
                "Agents and tasks lists cannot be empty."
            )

        responses = []

        for task in tasks:
            for agent in self.agents:
                response = agent.run(task)
                self.conversation.add(
                    role=agent.agent_name,
                    content=response,
                )
                responses.append(response)

        return self._format_return()


class StarSwarm(BaseSwarm):
    """
    Implements a star swarm where a central agent processes all tasks, followed by others.
    """

    def __init__(
        self,
        agents: AgentListType,
        name: str = "StarSwarm",
        description: str = "A star swarm where a central agent processes all tasks, followed by others",
        output_type: str = "dict",
    ):
        """
        Initialize the StarSwarm.

        Args:
            agents: List of Agent objects or nested list of Agent objects
            name: Name of the swarm
            description: Description of the swarm's purpose
            output_type: Type of output format, one of 'dict', 'list', 'string', 'json', 'yaml', 'xml', etc.
        """
        super().__init__(agents, name, description, output_type)

    def run(self, tasks: List[str]) -> Union[Dict, List, str]:
        """
        Run the star swarm with the given tasks

        Args:
            tasks: List of tasks to be processed

        Returns:
            Union[Dict, List, str]: The conversation history in the requested format
        """
        if not self.agents or not tasks:
            raise ValueError(
                "Agents and tasks lists cannot be empty."
            )

        responses = []
        center_agent = self.agents[0]  # The central agent

        for task in tasks:
            # Central agent processes the task
            center_response = center_agent.run(task)
            self.conversation.add(
                role=center_agent.agent_name,
                content=center_response,
            )
            responses.append(center_response)

            # Other agents process the same task
            for agent in self.agents[1:]:
                response = agent.run(task)
                self.conversation.add(
                    role=agent.agent_name,
                    content=response,
                )
                responses.append(response)

        return self._format_return()


class MeshSwarm(BaseSwarm):
    """
    Implements a mesh swarm where agents work on tasks randomly from a task queue.
    """

    def __init__(
        self,
        agents: AgentListType,
        name: str = "MeshSwarm",
        description: str = "A mesh swarm where agents work on tasks randomly from a task queue",
        output_type: str = "dict",
    ):
        """
        Initialize the MeshSwarm.

        Args:
            agents: List of Agent objects or nested list of Agent objects
            name: Name of the swarm
            description: Description of the swarm's purpose
            output_type: Type of output format, one of 'dict', 'list', 'string', 'json', 'yaml', 'xml', etc.
        """
        super().__init__(agents, name, description, output_type)

    def run(self, tasks: List[str]) -> Union[Dict, List, str]:
        """
        Run the mesh swarm with the given tasks

        Args:
            tasks: List of tasks to be processed

        Returns:
            Union[Dict, List, str]: The conversation history in the requested format
        """
        if not self.agents or not tasks:
            raise ValueError(
                "Agents and tasks lists cannot be empty."
            )

        task_queue = tasks.copy()
        responses = []

        while task_queue:
            for agent in self.agents:
                if task_queue:
                    task = task_queue.pop(0)
                    response = agent.run(task)
                    self.conversation.add(
                        role=agent.agent_name,
                        content=response,
                    )
                    responses.append(response)

        return self._format_return()


class PyramidSwarm(BaseSwarm):
    """
    Implements a pyramid swarm where agents are arranged in a pyramid structure.
    """

    def __init__(
        self,
        agents: AgentListType,
        name: str = "PyramidSwarm",
        description: str = "A pyramid swarm where agents are arranged in a pyramid structure",
        output_type: str = "dict",
    ):
        """
        Initialize the PyramidSwarm.

        Args:
            agents: List of Agent objects or nested list of Agent objects
            name: Name of the swarm
            description: Description of the swarm's purpose
            output_type: Type of output format, one of 'dict', 'list', 'string', 'json', 'yaml', 'xml', etc.
        """
        super().__init__(agents, name, description, output_type)

    def run(self, tasks: List[str]) -> Union[Dict, List, str]:
        """
        Run the pyramid swarm with the given tasks

        Args:
            tasks: List of tasks to be processed

        Returns:
            Union[Dict, List, str]: The conversation history in the requested format
        """
        if not self.agents or not tasks:
            raise ValueError(
                "Agents and tasks lists cannot be empty."
            )

        tasks_copy = tasks.copy()
        responses = []

        levels = int(
            (-1 + (1 + 8 * len(self.agents)) ** 0.5) / 2
        )  # Number of levels in the pyramid

        for i in range(levels):
            for j in range(i + 1):
                if tasks_copy:
                    task = tasks_copy.pop(0)
                    agent_index = int(i * (i + 1) / 2 + j)
                    if agent_index < len(self.agents):
                        response = self.agents[agent_index].run(task)
                        self.conversation.add(
                            role=self.agents[agent_index].agent_name,
                            content=response,
                        )
                        responses.append(response)

        return self._format_return()


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
