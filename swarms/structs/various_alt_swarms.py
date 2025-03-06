import math
from typing import List, Union, Dict

from loguru import logger

from swarms.structs.agent import Agent
from swarms.structs.omni_agent_types import AgentListType
from swarms.structs.conversation import Conversation


# Base Swarm class that all other swarm types will inherit from
class BaseSwarm:
    def __init__(self, agents: AgentListType):
        # Ensure agents is a flat list of Agent objects
        self.agents = (
            [agent for sublist in agents for agent in sublist]
            if isinstance(agents[0], list)
            else agents
        )
        self.conversation = Conversation()

    def run(
        self, tasks: List[str], return_type: str = "dict"
    ) -> Union[Dict, List, str]:
        """
        Run the swarm with the given tasks

        Args:
            tasks: List of tasks to be processed
            return_type: Type of return value, one of 'dict', 'list', or 'string'

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

    def _format_return(
        self, return_type: str
    ) -> Union[Dict, List, str]:
        """Format the return value based on the return_type"""
        if return_type.lower() == "dict":
            return self.conversation.return_messages_as_dictionary()
        elif return_type.lower() == "list":
            return self.conversation.return_messages_as_list()
        elif return_type.lower() == "string":
            return self.conversation.return_history_as_string()
        else:
            raise ValueError(
                "return_type must be one of 'dict', 'list', or 'string'"
            )


class CircularSwarm(BaseSwarm):
    """
    Implements a circular swarm where agents pass tasks in a circular manner.
    """

    def run(
        self, tasks: List[str], return_type: str = "dict"
    ) -> Union[Dict, List, str]:
        """
        Run the circular swarm with the given tasks

        Args:
            tasks: List of tasks to be processed
            return_type: Type of return value, one of 'dict', 'list', or 'string'

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

        return self._format_return(return_type)


class LinearSwarm(BaseSwarm):
    """
    Implements a linear swarm where agents process tasks sequentially.
    """

    def run(
        self, tasks: List[str], return_type: str = "dict"
    ) -> Union[Dict, List, str]:
        """
        Run the linear swarm with the given tasks

        Args:
            tasks: List of tasks to be processed
            return_type: Type of return value, one of 'dict', 'list', or 'string'

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

        return self._format_return(return_type)


class StarSwarm(BaseSwarm):
    """
    Implements a star swarm where a central agent processes all tasks, followed by others.
    """

    def run(
        self, tasks: List[str], return_type: str = "dict"
    ) -> Union[Dict, List, str]:
        """
        Run the star swarm with the given tasks

        Args:
            tasks: List of tasks to be processed
            return_type: Type of return value, one of 'dict', 'list', or 'string'

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

        return self._format_return(return_type)


class MeshSwarm(BaseSwarm):
    """
    Implements a mesh swarm where agents work on tasks randomly from a task queue.
    """

    def run(
        self, tasks: List[str], return_type: str = "dict"
    ) -> Union[Dict, List, str]:
        """
        Run the mesh swarm with the given tasks

        Args:
            tasks: List of tasks to be processed
            return_type: Type of return value, one of 'dict', 'list', or 'string'

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

        return self._format_return(return_type)


class PyramidSwarm(BaseSwarm):
    """
    Implements a pyramid swarm where agents are arranged in a pyramid structure.
    """

    def run(
        self, tasks: List[str], return_type: str = "dict"
    ) -> Union[Dict, List, str]:
        """
        Run the pyramid swarm with the given tasks

        Args:
            tasks: List of tasks to be processed
            return_type: Type of return value, one of 'dict', 'list', or 'string'

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

        return self._format_return(return_type)


class FibonacciSwarm(BaseSwarm):
    """
    Implements a Fibonacci swarm where agents are arranged according to the Fibonacci sequence.
    """

    def run(
        self, tasks: List[str], return_type: str = "dict"
    ) -> Union[Dict, List, str]:
        """
        Run the Fibonacci swarm with the given tasks

        Args:
            tasks: List of tasks to be processed
            return_type: Type of return value, one of 'dict', 'list', or 'string'

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

        return self._format_return(return_type)


class PrimeSwarm(BaseSwarm):
    """
    Implements a Prime swarm where agents at prime indices process tasks.
    """

    def run(
        self, tasks: List[str], return_type: str = "dict"
    ) -> Union[Dict, List, str]:
        """
        Run the Prime swarm with the given tasks

        Args:
            tasks: List of tasks to be processed
            return_type: Type of return value, one of 'dict', 'list', or 'string'

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

        return self._format_return(return_type)


class PowerSwarm(BaseSwarm):
    """
    Implements a Power swarm where agents at power-of-2 indices process tasks.
    """

    def run(
        self, tasks: List[str], return_type: str = "dict"
    ) -> Union[Dict, List, str]:
        """
        Run the Power swarm with the given tasks

        Args:
            tasks: List of tasks to be processed
            return_type: Type of return value, one of 'dict', 'list', or 'string'

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

        return self._format_return(return_type)


class LogSwarm(BaseSwarm):
    """
    Implements a Log swarm where agents at logarithmic indices process tasks.
    """

    def run(
        self, tasks: List[str], return_type: str = "dict"
    ) -> Union[Dict, List, str]:
        """
        Run the Log swarm with the given tasks

        Args:
            tasks: List of tasks to be processed
            return_type: Type of return value, one of 'dict', 'list', or 'string'

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

        return self._format_return(return_type)


class ExponentialSwarm(BaseSwarm):
    """
    Implements an Exponential swarm where agents at exponential indices process tasks.
    """

    def run(
        self, tasks: List[str], return_type: str = "dict"
    ) -> Union[Dict, List, str]:
        """
        Run the Exponential swarm with the given tasks

        Args:
            tasks: List of tasks to be processed
            return_type: Type of return value, one of 'dict', 'list', or 'string'

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

        return self._format_return(return_type)


class GeometricSwarm(BaseSwarm):
    """
    Implements a Geometric swarm where agents at geometrically increasing indices process tasks.
    """

    def run(
        self, tasks: List[str], return_type: str = "dict"
    ) -> Union[Dict, List, str]:
        """
        Run the Geometric swarm with the given tasks

        Args:
            tasks: List of tasks to be processed
            return_type: Type of return value, one of 'dict', 'list', or 'string'

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

        return self._format_return(return_type)


class HarmonicSwarm(BaseSwarm):
    """
    Implements a Harmonic swarm where agents at harmonically spaced indices process tasks.
    """

    def run(
        self, tasks: List[str], return_type: str = "dict"
    ) -> Union[Dict, List, str]:
        """
        Run the Harmonic swarm with the given tasks

        Args:
            tasks: List of tasks to be processed
            return_type: Type of return value, one of 'dict', 'list', or 'string'

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

        return self._format_return(return_type)


class StaircaseSwarm(BaseSwarm):
    """
    Implements a Staircase swarm where agents at staircase-patterned indices process a task.
    """

    def run(
        self, task: str, return_type: str = "dict"
    ) -> Union[Dict, List, str]:
        """
        Run the Staircase swarm with the given task

        Args:
            task: Task to be processed
            return_type: Type of return value, one of 'dict', 'list', or 'string'

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

        return self._format_return(return_type)


class SigmoidSwarm(BaseSwarm):
    """
    Implements a Sigmoid swarm where agents at sigmoid-distributed indices process a task.
    """

    def run(
        self, task: str, return_type: str = "dict"
    ) -> Union[Dict, List, str]:
        """
        Run the Sigmoid swarm with the given task

        Args:
            task: Task to be processed
            return_type: Type of return value, one of 'dict', 'list', or 'string'

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

        return self._format_return(return_type)


class SinusoidalSwarm(BaseSwarm):
    """
    Implements a Sinusoidal swarm where agents at sinusoidally-distributed indices process a task.
    """

    def run(
        self, task: str, return_type: str = "dict"
    ) -> Union[Dict, List, str]:
        """
        Run the Sinusoidal swarm with the given task

        Args:
            task: Task to be processed
            return_type: Type of return value, one of 'dict', 'list', or 'string'

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

        return self._format_return(return_type)


# Communication classes
class OneToOne:
    """
    Facilitates one-to-one communication between two agents.
    """

    def __init__(self, sender: Agent, receiver: Agent):
        self.sender = sender
        self.receiver = receiver
        self.conversation = Conversation()

    def run(
        self, task: str, max_loops: int = 1, return_type: str = "dict"
    ) -> Union[Dict, List, str]:
        """
        Run the one-to-one communication with the given task

        Args:
            task: Task to be processed
            max_loops: Number of exchange iterations
            return_type: Type of return value, one of 'dict', 'list', or 'string'

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

        if return_type.lower() == "dict":
            return self.conversation.return_messages_as_dictionary()
        elif return_type.lower() == "list":
            return self.conversation.return_messages_as_list()
        elif return_type.lower() == "string":
            return self.conversation.return_history_as_string()
        else:
            raise ValueError(
                "return_type must be one of 'dict', 'list', or 'string'"
            )


class Broadcast:
    """
    Facilitates broadcasting from one agent to many agents.
    """

    def __init__(self, sender: Agent, receivers: AgentListType):
        self.sender = sender
        self.receivers = (
            [agent for sublist in receivers for agent in sublist]
            if isinstance(receivers[0], list)
            else receivers
        )
        self.conversation = Conversation()

    def run(
        self, task: str, return_type: str = "dict"
    ) -> Union[Dict, List, str]:
        """
        Run the broadcast communication with the given task

        Args:
            task: Task to be processed
            return_type: Type of return value, one of 'dict', 'list', or 'string'

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

            if return_type.lower() == "dict":
                return (
                    self.conversation.return_messages_as_dictionary()
                )
            elif return_type.lower() == "list":
                return self.conversation.return_messages_as_list()
            elif return_type.lower() == "string":
                return self.conversation.return_history_as_string()
            else:
                raise ValueError(
                    "return_type must be one of 'dict', 'list', or 'string'"
                )

        except Exception as error:
            logger.error(f"Error during broadcast: {error}")
            raise error


class OneToThree:
    """
    Facilitates one-to-three communication from one agent to exactly three agents.
    """

    def __init__(self, sender: Agent, receivers: AgentListType):
        if len(receivers) != 3:
            raise ValueError(
                "The number of receivers must be exactly 3."
            )

        self.sender = sender
        self.receivers = receivers
        self.conversation = Conversation()

    def run(
        self, task: str, return_type: str = "dict"
    ) -> Union[Dict, List, str]:
        """
        Run the one-to-three communication with the given task

        Args:
            task: Task to be processed
            return_type: Type of return value, one of 'dict', 'list', or 'string'

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

            if return_type.lower() == "dict":
                return (
                    self.conversation.return_messages_as_dictionary()
                )
            elif return_type.lower() == "list":
                return self.conversation.return_messages_as_list()
            elif return_type.lower() == "string":
                return self.conversation.return_history_as_string()
            else:
                raise ValueError(
                    "return_type must be one of 'dict', 'list', or 'string'"
                )

        except Exception as error:
            logger.error(f"Error in one_to_three: {error}")
            raise error
