import random
from typing import List, Union

import tenacity

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType

logger = initialize_logger("round-robin")


class RoundRobinSwarm:
    """
    A swarm implementation that executes tasks in a round-robin fashion.

    This swarm implements an AutoGen-style communication pattern where agents
    are shuffled randomly each loop for varied interaction patterns. Each agent
    receives the full conversation context to build upon others' responses.

    Args:
        name (str): Name of the swarm. Defaults to "RoundRobinSwarm".
        description (str): Description of the swarm's purpose.
        agents (List[Agent]): List of agents in the swarm. Required.
        verbose (bool, optional): Flag to enable verbose mode. Defaults to False.
        max_loops (int, optional): Maximum number of loops to run. Defaults to 1.
        callback (callable, optional): Callback function to be called after each loop. Defaults to None.
        max_retries (int, optional): Maximum number of retries for agent execution. Defaults to 3.
        output_type (OutputType, optional): Type of output format. Defaults to "final".

    Attributes:
        name (str): Name of the swarm.
        description (str): Description of the swarm's purpose.
        agents (List[Agent]): List of agents in the swarm.
        verbose (bool): Flag to enable verbose mode.
        max_loops (int): Maximum number of loops to run.
        callback (callable): Callback function executed after each loop.
        index (int): Current index of the agent being executed.
        max_retries (int): Maximum number of retries for agent execution.
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
        callback: callable = None,
        max_retries: int = 3,
        output_type: OutputType = "final",
    ):

        self.name = name
        self.description = description
        self.agents = agents
        self.verbose = verbose
        self.max_loops = max_loops
        self.callback = callback
        self.index = 0
        self.max_retries = max_retries
        self.output_type = output_type

        # Initialize conversation for tracking agent interactions
        self.conversation = Conversation(name=f"{name}_conversation")

        if self.agents is None:
            raise ValueError(
                "RoundRobinSwarm cannot be initialized without agents"
            )

        # Set the max loops for every agent
        if self.agents:
            for agent in self.agents:
                agent.max_loops = random.randint(1, 5)

        logger.info(
            f"Successfully initialized {self.name} with {len(self.agents)} agents"
        )

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        retry=tenacity.retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.info(
            f"Retrying in {retry_state.next_action.sleep} seconds..."
        ),
    )
    def _execute_agent(
        self, agent: Agent, task: str, *args, **kwargs
    ) -> str:
        """
        Execute a single agent with retries and error handling.

        Args:
            agent (Agent): The agent to execute.
            task (str): The task to be executed.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            str: The result of the agent execution.
        """
        try:
            logger.info(
                f"Running Agent {agent.agent_name} on task: {task}"
            )
            result = agent.run(task, *args, **kwargs)
            self.conversation.add(
                role=agent.agent_name,
                content=result,
            )
            return result
        except Exception as e:
            logger.error(
                f"Error executing agent {agent.agent_name}: {str(e)}"
            )
            raise

    def run(
        self, task: str, *args, **kwargs
    ) -> Union[str, dict, list]:
        """
        Executes the given task on the agents in a randomized round-robin fashion.

        This method implements an AutoGen-style communication pattern where:
        - Agents are shuffled randomly each loop for varied interaction patterns
        - Each agent receives the full conversation context to build upon others' responses
        - Collaborative prompting encourages agents to acknowledge and extend prior contributions

        Args:
            task (str): The task to be executed.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Union[str, dict, list]: The result of the task execution in the specified output format.

        Raises:
            ValueError: If no agents are configured
            Exception: If an exception occurs during task execution.
        """
        if not self.agents:
            logger.error("No agents configured for the swarm")
            raise ValueError("No agents configured for the swarm")

        try:
            # Add initial task to conversation
            self.conversation.add(role="User", content=task)
            n = len(self.agents)

            # Build agent names list for context
            agent_names = [agent.agent_name for agent in self.agents]

            logger.info(
                f"Starting randomized round-robin execution with task on {n} agents: {agent_names}"
            )

            for loop in range(self.max_loops):
                logger.debug(
                    f"Starting loop {loop + 1}/{self.max_loops}"
                )

                # Shuffle agents randomly each loop for varied interaction patterns
                shuffled_agents = self.agents.copy()
                random.shuffle(shuffled_agents)

                logger.debug(
                    f"Agent order for loop {loop + 1}: {[a.agent_name for a in shuffled_agents]}"
                )

                for i, current_agent in enumerate(shuffled_agents):
                    # Get current conversation context
                    conversation_context = (
                        self.conversation.return_history_as_string()
                    )

                    # Build collaborative prompt with context
                    collaborative_task = f"""{conversation_context}

                    As {current_agent.agent_name}, you are agent {i + 1} of {n} in this collaborative session. The other agents participating are: {', '.join(name for name in agent_names if name != current_agent.agent_name)}.

                    Please review the conversation history above carefully and build upon the insights shared by other agents. Acknowledge their contributions where relevant and provide your unique perspective and expertise. Be concise but thorough in your response, and if this is the first response in the conversation, address the original task directly.

                    Your response:"""

                    try:
                        result = self._execute_agent(
                            current_agent,
                            collaborative_task,
                            *args,
                            **kwargs,
                        )
                    except Exception as e:
                        logger.error(
                            f"Agent {current_agent.agent_name} failed: {str(e)}"
                        )
                        raise

                if self.callback:
                    logger.debug(
                        f"Executing callback for loop {loop + 1}"
                    )
                    try:
                        self.callback(loop, result)
                    except Exception as e:
                        logger.error(
                            f"Callback execution failed: {str(e)}"
                        )

            logger.success(
                f"Successfully completed {self.max_loops} loops of randomized round-robin execution"
            )

            return history_output_formatter(
                conversation=self.conversation,
                type=self.output_type,
            )

        except Exception as e:
            logger.error(f"Round-robin execution failed: {str(e)}")
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
