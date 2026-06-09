from typing import List, Union

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType
from swarms.structs.ma_blocks import return_all_agent_names
from swarms.structs.serialization import SerializableMixin

logger = initialize_logger("round-robin")


def build_turn_header(
    agent_name: str,
    position: int,
    total: int,
    loop: int,
    max_loops: int,
    prev_name: str = None,
    next_name: str = None,
    agent_names: List[str] = None,
) -> str:
    """Build the per-turn role header injected above each agent's task.

    Args:
        agent_name: Name of the agent taking this turn.
        position: 1-indexed position of this agent within the current loop.
        total: Total number of agents in the roster.
        loop: 1-indexed loop number.
        max_loops: Total number of loops the swarm will run.
        prev_name: Name of the agent who spoke immediately before, or
            None if this is the opening turn.
        next_name: Name of the agent who will speak next, or None if
            this is the closing turn.
        agent_names: All agent names in the swarm, used to populate the
            "Other participants" list.
    """
    others = (
        ", ".join(n for n in (agent_names or []) if n != agent_name)
        or "(none)"
    )
    return (
        f"You are {agent_name}, "
        f"agent {position} of {total} in loop {loop} of {max_loops}. "
        f"Previous speaker: {prev_name or '(none — you open the conversation)'}. "
        f"Next speaker: {next_name or '(none — you close the conversation)'}. "
        f"Other participants: {others}."
    )


def build_collaborative_task(
    conversation_context: str, turn_header: str
) -> str:
    """Build the full prompt passed to an agent on its turn.

    Concatenates the running transcript with the per-turn role header and
    the standing collaboration instruction.
    """
    return (
        f"{conversation_context}\n\n"
        f"{turn_header}\n\n"
        "Review the transcript above and build on the prior speaker's contribution. "
        "Add your own perspective concisely; if you are the opening speaker, address the original task directly.\n\n"
        "Your response:"
    )


class RoundRobinSwarm(SerializableMixin):
    """
    A swarm implementation that executes tasks in a true round-robin fashion.

    Agents are visited in their declared insertion order, cycling through the
    full roster once per loop. Over K loops with N agents the schedule is:

        turn t -> agents[t % N]    for t in range(K * N)

    The order is deterministic and identical on every loop, so each agent
    receives exactly `max_loops` turns and every agent reads the full
    conversation history accumulated by the agents that spoke before it.

    Args:
        name (str): Name of the swarm. Defaults to "RoundRobinSwarm".
        description (str): Description of the swarm's purpose.
        agents (List[Agent]): List of agents in the swarm. Required.
        verbose (bool, optional): Flag to enable verbose mode. Defaults to False.
        max_loops (int, optional): Maximum number of loops to run. Defaults to 1.
        output_type (OutputType, optional): Type of output format. Defaults to "final".

    Attributes:
        name (str): Name of the swarm.
        description (str): Description of the swarm's purpose.
        agents (List[Agent]): List of agents in the swarm.
        verbose (bool): Flag to enable verbose mode.
        max_loops (int): Maximum number of loops to run.
        index (int): Current index of the agent being executed.
        output_type (OutputType): Type of output format.
        conversation (Conversation): Conversation history for the swarm.

    Methods:
        run(task: str, *args, **kwargs) -> Union[str, dict, list]:
            Executes the given task on the agents in a round-robin fashion.
        run_batch(tasks: List[str]) -> List:
            Executes multiple tasks sequentially, returning results for each.

    Raises:
        ValueError: If no agents are provided during initialization.

    """

    def __init__(
        self,
        name: str = "RoundRobinSwarm",
        description: str = "A swarm implementation that executes tasks in a round-robin fashion.",
        agents: List[Agent] = None,
        verbose: bool = False,
        max_loops: int = 1,
        output_type: OutputType = "final",
    ):

        self.name = name
        self.description = description
        self.agents = agents
        self.verbose = verbose
        self.max_loops = max_loops
        self.index = 0
        self.output_type = output_type

        # Initialize conversation for tracking agent interactions
        self.conversation = Conversation(
            name=f"{name}_conversation",
            time_enabled=True,
            message_id_on=True,
        )

        if not self.agents:
            raise ValueError(
                "RoundRobinSwarm cannot be initialized without agents"
            )

        self._log(
            "info",
            f"Successfully initialized {self.name} with {len(self.agents)} agents",
        )

    def _execute_agent(
        self, agent: Agent, task: str, *args, **kwargs
    ) -> str:
        """
        Execute a single agent and append its response to the conversation.

        Args:
            agent (Agent): The agent to execute.
            task (str): The task to be executed.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            str: The result of the agent execution.
        """
        try:
            self._log(
                "info",
                f"Running Agent {agent.agent_name} on task: {task}",
            )
            result = agent.run(task, *args, **kwargs)
            self.conversation.add(
                role=agent.agent_name,
                content=result,
            )
            return result
        except Exception as e:
            self._log(
                "error",
                f"Error executing agent {agent.agent_name}: {str(e)}",
            )
            raise

    def run(
        self, task: str, *args, **kwargs
    ) -> Union[str, dict, list]:
        """
        Execute the task across the agents in true round-robin order.

        The schedule is deterministic: for N agents and `max_loops` loops the
        visit order is `agents[t % N]` for `t` in `range(max_loops * N)`. The
        order is identical on every loop, every agent reads the full
        conversation transcript accumulated so far, and every agent receives
        exactly `max_loops` turns.

        Args:
            task (str): The task to be executed.
            *args: Variable length argument list passed to each agent.
            **kwargs: Arbitrary keyword arguments passed to each agent.

        Returns:
            Union[str, dict, list]: The result of the task execution in the
                format specified by `output_type`.

        Raises:
            Exception: If an exception occurs during task execution.
        """
        try:
            self.conversation.add(role="User", content=task)
            n = len(self.agents)
            agent_names = return_all_agent_names(self.agents)

            self._log(
                "info",
                f"Starting round-robin execution with task on {n} agents: {agent_names}",
            )

            for loop in range(self.max_loops):
                self._log(
                    "debug",
                    f"Starting loop {loop + 1}/{self.max_loops}",
                )

                for i, current_agent in enumerate(self.agents):
                    self.index = (loop * n) + i

                    prev_name = (
                        self.agents[i - 1].agent_name
                        if i > 0
                        else (
                            self.agents[-1].agent_name
                            if loop > 0
                            else None
                        )
                    )
                    next_name = (
                        self.agents[i + 1].agent_name
                        if i + 1 < n
                        else (
                            self.agents[0].agent_name
                            if loop + 1 < self.max_loops
                            else None
                        )
                    )

                    conversation_context = (
                        self.conversation.return_history_as_string()
                    )

                    turn_header = build_turn_header(
                        agent_name=current_agent.agent_name,
                        position=i + 1,
                        total=n,
                        loop=loop + 1,
                        max_loops=self.max_loops,
                        prev_name=prev_name,
                        next_name=next_name,
                        agent_names=agent_names,
                    )

                    collaborative_task = build_collaborative_task(
                        conversation_context=conversation_context,
                        turn_header=turn_header,
                    )

                    try:
                        self._execute_agent(
                            current_agent,
                            collaborative_task,
                            *args,
                            **kwargs,
                        )
                    except Exception as e:
                        self._log(
                            "error",
                            f"Agent {current_agent.agent_name} failed: {str(e)}",
                        )
                        raise

            self._log(
                "success",
                f"Successfully completed {self.max_loops} loops of round-robin execution",
            )

            return history_output_formatter(
                conversation=self.conversation,
                type=self.output_type,
            )

        except Exception as e:
            self._log(
                "error", f"Round-robin execution failed: {str(e)}"
            )
            raise

    def run_batch(
        self, tasks: List[str]
    ) -> List[Union[str, dict, list]]:
        """
        Execute multiple tasks sequentially through the round-robin swarm.

        Each task is processed independently through the full round-robin
        execution cycle, with agents collaborating on each task in turn.

        Args:
            tasks (List[str]): A list of task strings to be executed.

        Returns:
            List[Union[str, dict, list]]: A list of results, one for each task,
                in the format specified by output_type.

        Example:
            >>> swarm = RoundRobinSwarm(agents=[agent1, agent2])
            >>> results = swarm.run_batch(["Task 1", "Task 2", "Task 3"])
        """
        return [self.run(task) for task in tasks]
