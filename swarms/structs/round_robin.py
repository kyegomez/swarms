from typing import List, Union

from swarms.structs.execution_utils import batched_run
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType
from swarms.structs.ma_blocks import return_all_agent_names
from swarms.structs.serialization import SerializableMixin
from swarms.telemetry.otel import capture_init, trace_run

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

    Agents are visited in roster order, cycling through the full list once
    per loop. Over K loops with N agents the within-run schedule is:

        turn t -> agents[(start + t) % N]    for t in range(K * N)

    ``start`` is ``self.index`` when ``persist_rotation`` is True, otherwise
    0. By default ``persist_rotation`` is False so each ``run()`` replays
    the same visit order turn-for-turn; set it to True to resume rotation
    across ``run()`` / ``run_batch()`` calls.

    The order is deterministic within a run: each agent receives exactly
    `max_loops` turns and every agent reads the full conversation history
    accumulated by the agents that spoke before it.

    Args:
        name (str): Name of the swarm. Defaults to "RoundRobinSwarm".
        description (str): Description of the swarm's purpose.
        agents (List[Agent]): List of agents in the swarm. Required.
        verbose (bool, optional): Flag to enable verbose mode. Defaults to False.
        max_loops (int, optional): Maximum number of loops to run. Defaults to 1.
        output_type (OutputType, optional): Type of output format. Defaults to "final".
        persist_rotation (bool, optional): When True, resume the round-robin
            pointer across ``run()`` calls so ``run_batch`` gives each agent
            the opening turn equally often. Defaults to False to preserve
            current turn-for-turn reproducibility.

    Attributes:
        name (str): Name of the swarm.
        description (str): Description of the swarm's purpose.
        agents (List[Agent]): List of agents in the swarm.
        verbose (bool): Flag to enable verbose mode.
        max_loops (int): Maximum number of loops to run.
        index (int): Roster offset for the next run's opening turn when
            ``persist_rotation`` is True.
        persist_rotation (bool): Whether rotation carries across ``run()`` calls.
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
        persist_rotation: bool = False,
    ):

        self.name = name
        self.description = description
        self.agents = agents
        self.verbose = verbose
        self.max_loops = max_loops
        self.index = 0
        self.output_type = output_type
        self.persist_rotation = persist_rotation

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

        # Capture the full __init__ configuration if telemetry is enabled.
        capture_init(self)

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

    @trace_run(
        "RoundRobinSwarm.run",
        input_params=("task", "tasks", "img", "imgs"),
    )
    def run(
        self, task: str, *args, **kwargs
    ) -> Union[str, dict, list]:
        """
        Execute the task across the agents in true round-robin order.

        The schedule is deterministic within a run: for N agents and
        ``max_loops`` loops the visit order is
        ``agents[(start + t) % N]`` for ``t`` in ``range(max_loops * N)``,
        where ``start`` is ``self.index`` when ``persist_rotation`` is True.
        Every agent receives exactly ``max_loops`` turns. When rotation
        persists, the opening turn advances by one after each run so
        ``run_batch`` does not always seat ``agents[0]`` first.

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

            start = self.index % n if self.persist_rotation else 0
            total_turns = self.max_loops * n

            for turn in range(total_turns):
                loop = turn // n + 1
                roster_i = (start + turn) % n
                current_agent = self.agents[roster_i]

                if turn % n == 0:
                    self._log(
                        "debug",
                        f"Starting loop {loop}/{self.max_loops}",
                    )

                prev_name = (
                    None
                    if turn == 0
                    else self.agents[
                        (start + turn - 1) % n
                    ].agent_name
                )
                next_name = (
                    None
                    if turn == total_turns - 1
                    else self.agents[
                        (start + turn + 1) % n
                    ].agent_name
                )

                conversation_context = (
                    self.conversation.return_history_as_string()
                )

                turn_header = build_turn_header(
                    agent_name=current_agent.agent_name,
                    position=turn % n + 1,
                    total=n,
                    loop=loop,
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

            # max_loops * n turns is a whole number of cycles, so it returns
            # the offset to where it started — advancing the opener needs +1.
            if self.persist_rotation:
                self.index = (start + 1) % n

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
        return batched_run(self.run, tasks)
