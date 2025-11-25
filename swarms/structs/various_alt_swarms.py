import math
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


class LinearSwarm(BaseSwarm):
    """
    Implements a linear swarm where agents process tasks sequentially.
    """

    def __init__(
        self,
        agents: AgentListType,
        name: str = "LinearSwarm",
        description: str = "A linear swarm where agents process tasks sequentially",
        output_type: str = "dict",
    ):
        """
        Initialize the LinearSwarm.

        Args:
            agents: List of Agent objects or nested list of Agent objects
            name: Name of the swarm
            description: Description of the swarm's purpose
            output_type: Type of output format, one of 'dict', 'list', 'string', 'json', 'yaml', 'xml', etc.
        """
        super().__init__(agents, name, description, output_type)

    def run(self, tasks: List[str]) -> Union[Dict, List, str]:
        """
        Run the linear swarm with the given tasks

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

        for agent in self.agents:
            if tasks_copy:
                task = tasks_copy.pop(0)
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


class FibonacciSwarm(BaseSwarm):
    """
    Implements a Fibonacci swarm where agents are arranged according to the Fibonacci sequence.
    """

    def __init__(
        self,
        agents: AgentListType,
        name: str = "FibonacciSwarm",
        description: str = "A Fibonacci swarm where agents are arranged according to the Fibonacci sequence",
        output_type: str = "dict",
    ):
        """
        Initialize the FibonacciSwarm.

        Args:
            agents: List of Agent objects or nested list of Agent objects
            name: Name of the swarm
            description: Description of the swarm's purpose
            output_type: Type of output format, one of 'dict', 'list', 'string', 'json', 'yaml', 'xml', etc.
        """
        super().__init__(agents, name, description, output_type)

    def run(self, tasks: List[str]) -> Union[Dict, List, str]:
        """
        Run the Fibonacci swarm with the given tasks

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

        fib = [1, 1]
        while len(fib) < len(self.agents):
            fib.append(fib[-1] + fib[-2])

        for i in range(len(fib)):
            for j in range(fib[i]):
                agent_index = int(sum(fib[:i]) + j)
                if agent_index < len(self.agents) and tasks_copy:
                    task = tasks_copy.pop(0)
                    response = self.agents[agent_index].run(task)
                    self.conversation.add(
                        role=self.agents[agent_index].agent_name,
                        content=response,
                    )
                    responses.append(response)

        return self._format_return()


class PrimeSwarm(BaseSwarm):
    """
    Implements a Prime swarm where agents at prime indices process tasks.
    """

    def __init__(
        self,
        agents: AgentListType,
        name: str = "PrimeSwarm",
        description: str = "A Prime swarm where agents at prime indices process tasks",
        output_type: str = "dict",
    ):
        """
        Initialize the PrimeSwarm.

        Args:
            agents: List of Agent objects or nested list of Agent objects
            name: Name of the swarm
            description: Description of the swarm's purpose
            output_type: Type of output format, one of 'dict', 'list', 'string', 'json', 'yaml', 'xml', etc.
        """
        super().__init__(agents, name, description, output_type)

    def run(self, tasks: List[str]) -> Union[Dict, List, str]:
        """
        Run the Prime swarm with the given tasks

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
            if prime < len(self.agents) and tasks_copy:
                task = tasks_copy.pop(0)
                response = self.agents[prime].run(task)
                self.conversation.add(
                    role=self.agents[prime].agent_name,
                    content=response,
                )
                responses.append(response)

        return self._format_return()


class PowerSwarm(BaseSwarm):
    """
    Implements a Power swarm where agents at power-of-2 indices process tasks.
    """

    def __init__(
        self,
        agents: AgentListType,
        name: str = "PowerSwarm",
        description: str = "A Power swarm where agents at power-of-2 indices process tasks",
        output_type: str = "dict",
    ):
        """
        Initialize the PowerSwarm.

        Args:
            agents: List of Agent objects or nested list of Agent objects
            name: Name of the swarm
            description: Description of the swarm's purpose
            output_type: Type of output format, one of 'dict', 'list', 'string', 'json', 'yaml', 'xml', etc.
        """
        super().__init__(agents, name, description, output_type)

    def run(self, tasks: List[str]) -> Union[Dict, List, str]:
        """
        Run the Power swarm with the given tasks

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

        powers = [2**i for i in range(int(len(self.agents) ** 0.5))]

        for power in powers:
            if power < len(self.agents) and tasks_copy:
                task = tasks_copy.pop(0)
                response = self.agents[power].run(task)
                self.conversation.add(
                    role=self.agents[power].agent_name,
                    content=response,
                )
                responses.append(response)

        return self._format_return()


class LogSwarm(BaseSwarm):
    """
    Implements a Log swarm where agents at logarithmic indices process tasks.
    """

    def __init__(
        self,
        agents: AgentListType,
        name: str = "LogSwarm",
        description: str = "A Log swarm where agents at logarithmic indices process tasks",
        output_type: str = "dict",
    ):
        """
        Initialize the LogSwarm.

        Args:
            agents: List of Agent objects or nested list of Agent objects
            name: Name of the swarm
            description: Description of the swarm's purpose
            output_type: Type of output format, one of 'dict', 'list', 'string', 'json', 'yaml', 'xml', etc.
        """
        super().__init__(agents, name, description, output_type)

    def run(self, tasks: List[str]) -> Union[Dict, List, str]:
        """
        Run the Log swarm with the given tasks

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

        for i in range(len(self.agents)):
            index = 2**i
            if index < len(self.agents) and tasks_copy:
                task = tasks_copy.pop(0)
                response = self.agents[index].run(task)
                self.conversation.add(
                    role=self.agents[index].agent_name,
                    content=response,
                )
                responses.append(response)

        return self._format_return()


class ExponentialSwarm(BaseSwarm):
    """
    Implements an Exponential swarm where agents at exponential indices process tasks.
    """

    def __init__(
        self,
        agents: AgentListType,
        name: str = "ExponentialSwarm",
        description: str = "An Exponential swarm where agents at exponential indices process tasks",
        output_type: str = "dict",
    ):
        """
        Initialize the ExponentialSwarm.

        Args:
            agents: List of Agent objects or nested list of Agent objects
            name: Name of the swarm
            description: Description of the swarm's purpose
            output_type: Type of output format, one of 'dict', 'list', 'string', 'json', 'yaml', 'xml', etc.
        """
        super().__init__(agents, name, description, output_type)

    def run(self, tasks: List[str]) -> Union[Dict, List, str]:
        """
        Run the Exponential swarm with the given tasks

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

        for i in range(len(self.agents)):
            index = min(int(2**i), len(self.agents) - 1)
            if tasks_copy:
                task = tasks_copy.pop(0)
                response = self.agents[index].run(task)
                self.conversation.add(
                    role=self.agents[index].agent_name,
                    content=response,
                )
                responses.append(response)

        return self._format_return()


class GeometricSwarm(BaseSwarm):
    """
    Implements a Geometric swarm where agents at geometrically increasing indices process tasks.
    """

    def __init__(
        self,
        agents: AgentListType,
        name: str = "GeometricSwarm",
        description: str = "A Geometric swarm where agents at geometrically increasing indices process tasks",
        output_type: str = "dict",
    ):
        """
        Initialize the GeometricSwarm.

        Args:
            agents: List of Agent objects or nested list of Agent objects
            name: Name of the swarm
            description: Description of the swarm's purpose
            output_type: Type of output format, one of 'dict', 'list', 'string', 'json', 'yaml', 'xml', etc.
        """
        super().__init__(agents, name, description, output_type)

    def run(self, tasks: List[str]) -> Union[Dict, List, str]:
        """
        Run the Geometric swarm with the given tasks

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
        ratio = 2

        for i in range(len(self.agents)):
            index = min(int(ratio**i), len(self.agents) - 1)
            if tasks_copy:
                task = tasks_copy.pop(0)
                response = self.agents[index].run(task)
                self.conversation.add(
                    role=self.agents[index].agent_name,
                    content=response,
                )
                responses.append(response)

        return self._format_return()


class HarmonicSwarm(BaseSwarm):
    """
    Implements a Harmonic swarm where agents at harmonically spaced indices process tasks.
    """

    def __init__(
        self,
        agents: AgentListType,
        name: str = "HarmonicSwarm",
        description: str = "A Harmonic swarm where agents at harmonically spaced indices process tasks",
        output_type: str = "dict",
    ):
        """
        Initialize the HarmonicSwarm.

        Args:
            agents: List of Agent objects or nested list of Agent objects
            name: Name of the swarm
            description: Description of the swarm's purpose
            output_type: Type of output format, one of 'dict', 'list', 'string', 'json', 'yaml', 'xml', etc.
        """
        super().__init__(agents, name, description, output_type)

    def run(self, tasks: List[str]) -> Union[Dict, List, str]:
        """
        Run the Harmonic swarm with the given tasks

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

        for i in range(1, len(self.agents) + 1):
            index = min(
                int(len(self.agents) / i), len(self.agents) - 1
            )
            if tasks_copy:
                task = tasks_copy.pop(0)
                response = self.agents[index].run(task)
                self.conversation.add(
                    role=self.agents[index].agent_name,
                    content=response,
                )
                responses.append(response)

        return self._format_return()


class StaircaseSwarm(BaseSwarm):
    """
    Implements a Staircase swarm where agents at staircase-patterned indices process a task.
    """

    def __init__(
        self,
        agents: AgentListType,
        name: str = "StaircaseSwarm",
        description: str = "A Staircase swarm where agents at staircase-patterned indices process a task",
        output_type: str = "dict",
    ):
        """
        Initialize the StaircaseSwarm.

        Args:
            agents: List of Agent objects or nested list of Agent objects
            name: Name of the swarm
            description: Description of the swarm's purpose
            output_type: Type of output format, one of 'dict', 'list', 'string', 'json', 'yaml', 'xml', etc.
        """
        super().__init__(agents, name, description, output_type)

    def run(self, task: str) -> Union[Dict, List, str]:
        """
        Run the Staircase swarm with the given task

        Args:
            task: Task to be processed

        Returns:
            Union[Dict, List, str]: The conversation history in the requested format
        """
        if not self.agents or not task:
            raise ValueError("Agents and task cannot be empty.")

        responses = []
        step = len(self.agents) // 5

        for i in range(len(self.agents)):
            index = (i // step) * step
            if index < len(self.agents):
                response = self.agents[index].run(task)
                self.conversation.add(
                    role=self.agents[index].agent_name,
                    content=response,
                )
                responses.append(response)

        return self._format_return()


class SigmoidSwarm(BaseSwarm):
    """
    Implements a Sigmoid swarm where agents at sigmoid-distributed indices process a task.
    """

    def __init__(
        self,
        agents: AgentListType,
        name: str = "SigmoidSwarm",
        description: str = "A Sigmoid swarm where agents at sigmoid-distributed indices process a task",
        output_type: str = "dict",
    ):
        """
        Initialize the SigmoidSwarm.

        Args:
            agents: List of Agent objects or nested list of Agent objects
            name: Name of the swarm
            description: Description of the swarm's purpose
            output_type: Type of output format, one of 'dict', 'list', 'string', 'json', 'yaml', 'xml', etc.
        """
        super().__init__(agents, name, description, output_type)

    def run(self, task: str) -> Union[Dict, List, str]:
        """
        Run the Sigmoid swarm with the given task

        Args:
            task: Task to be processed

        Returns:
            Union[Dict, List, str]: The conversation history in the requested format
        """
        if not self.agents or not task:
            raise ValueError("Agents and task cannot be empty.")

        responses = []

        for i in range(len(self.agents)):
            index = int(len(self.agents) / (1 + math.exp(-i)))
            if index < len(self.agents):
                response = self.agents[index].run(task)
                self.conversation.add(
                    role=self.agents[index].agent_name,
                    content=response,
                )
                responses.append(response)

        return self._format_return()


class SinusoidalSwarm(BaseSwarm):
    """
    Implements a Sinusoidal swarm where agents at sinusoidally-distributed indices process a task.
    """

    def __init__(
        self,
        agents: AgentListType,
        name: str = "SinusoidalSwarm",
        description: str = "A Sinusoidal swarm where agents at sinusoidally-distributed indices process a task",
        output_type: str = "dict",
    ):
        """
        Initialize the SinusoidalSwarm.

        Args:
            agents: List of Agent objects or nested list of Agent objects
            name: Name of the swarm
            description: Description of the swarm's purpose
            output_type: Type of output format, one of 'dict', 'list', 'string', 'json', 'yaml', 'xml', etc.
        """
        super().__init__(agents, name, description, output_type)

    def run(self, task: str) -> Union[Dict, List, str]:
        """
        Run the Sinusoidal swarm with the given task

        Args:
            task: Task to be processed

        Returns:
            Union[Dict, List, str]: The conversation history in the requested format
        """
        if not self.agents or not task:
            raise ValueError("Agents and task cannot be empty.")

        responses = []

        for i in range(len(self.agents)):
            index = int((math.sin(i) + 1) / 2 * len(self.agents))
            if index < len(self.agents):
                response = self.agents[index].run(task)
                self.conversation.add(
                    role=self.agents[index].agent_name,
                    content=response,
                )
                responses.append(response)

        return self._format_return()


# Communication classes
class OneToOne:
    """
    Facilitates one-to-one communication between two agents.
    """

    def __init__(
        self,
        sender: Agent,
        receiver: Agent,
        output_type: str = "dict",
    ):
        """
        Initialize the OneToOne communication.

        Args:
            sender: The sender agent
            receiver: The receiver agent
            output_type: Type of output format, one of 'dict', 'list', 'string', 'json', 'yaml', 'xml', etc.
        """
        self.sender = sender
        self.receiver = receiver
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
        output_type: str = "dict",
    ):
        """
        Initialize the Broadcast communication.

        Args:
            sender: The sender agent
            receivers: List of receiver agents
            output_type: Type of output format, one of 'dict', 'list', 'string', 'json', 'yaml', 'xml', etc.
        """
        self.sender = sender
        self.receivers = (
            [agent for sublist in receivers for agent in sublist]
            if isinstance(receivers[0], list)
            else receivers
        )
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
        output_type: str = "dict",
    ):
        """
        Initialize the OneToThree communication.

        Args:
            sender: The sender agent
            receivers: List of exactly three receiver agents
            output_type: Type of output format, one of 'dict', 'list', 'string', 'json', 'yaml', 'xml', etc.
        """
        if len(receivers) != 3:
            raise ValueError(
                "The number of receivers must be exactly 3."
            )

        self.sender = sender
        self.receivers = receivers
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
            logger.error(f"Error in one_to_three: {error}")
            raise error
