from typing import List
from swarms.structs.agent import Agent
from swarms.utils.loguru_logger import logger
from swarms.structs.rearrange import AgentRearrange, OutputType
from concurrent.futures import ThreadPoolExecutor, as_completed
from swarms.structs.agents_available import showcase_available_agents


class SequentialWorkflow:
    """
    Initializes a SequentialWorkflow object, which orchestrates the execution of a sequence of agents.

    Args:
        name (str, optional): The name of the workflow. Defaults to "SequentialWorkflow".
        description (str, optional): A description of the workflow. Defaults to "Sequential Workflow, where agents are executed in a sequence."
        agents (List[Agent], optional): The list of agents in the workflow. Defaults to None.
        max_loops (int, optional): The maximum number of loops to execute the workflow. Defaults to 1.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Raises:
        ValueError: If agents list is None or empty, or if max_loops is 0
    """

    def __init__(
        self,
        name: str = "SequentialWorkflow",
        description: str = "Sequential Workflow, where agents are executed in a sequence.",
        agents: List[Agent] = [],
        max_loops: int = 1,
        output_type: OutputType = "all",
        return_json: bool = False,
        shared_memory_system: callable = None,
        *args,
        **kwargs,
    ):
        self.name = name
        self.description = description
        self.agents = agents
        self.max_loops = max_loops
        self.output_type = output_type
        self.return_json = return_json
        self.shared_memory_system = shared_memory_system

        self.reliability_check()

        self.agent_rearrange = AgentRearrange(
            name=name,
            description=description,
            agents=agents,
            flow=self.sequential_flow(),
            max_loops=max_loops,
            output_type=output_type,
            return_json=return_json,
            shared_memory_system=shared_memory_system,
            *args,
            **kwargs,
        )

        # Handle agent showcase
        self.handle_agent_showcase()

    def sequential_flow(self):
        # Only create flow if agents exist
        if self.agents:
            # Create flow by joining agent names with arrows
            agent_names = []
            for agent in self.agents:
                try:
                    # Try to get agent_name, fallback to name if not available
                    agent_name = (
                        getattr(agent, "agent_name", None)
                        or agent.name
                    )
                    agent_names.append(agent_name)
                except AttributeError:
                    logger.warning(
                        f"Could not get name for agent {agent}"
                    )
                    continue

            if agent_names:
                flow = " -> ".join(agent_names)
            else:
                flow = ""
                logger.warning(
                    "No valid agent names found to create flow"
                )
        else:
            flow = ""
            logger.warning("No agents provided to create flow")

        return flow

    def reliability_check(self):
        if self.agents is None or len(self.agents) == 0:
            raise ValueError("Agents list cannot be None or empty")

        if self.max_loops == 0:
            raise ValueError("max_loops cannot be 0")

        if self.output_type not in OutputType:
            raise ValueError(
                "output_type must be 'all', 'final', 'list', 'dict', '.json', '.md', '.txt', '.yaml', or '.toml'"
            )

        logger.info("Checks completed your swarm is ready.")

    def handle_agent_showcase(self):
        # Get the showcase string once instead of regenerating for each agent
        showcase_str = showcase_available_agents(
            name=self.name,
            description=self.description,
            agents=self.agents,
        )

        # Append showcase string to each agent's existing system prompt
        for agent in self.agents:
            agent.system_prompt += showcase_str

    def run(
        self,
        task: str,
        device: str = "cpu",
        all_cpus: bool = False,
        auto_gpu: bool = False,
        *args,
        **kwargs,
    ) -> str:
        """
        Executes a task through the agents in the dynamically constructed flow.

        Args:
            task (str): The task for the agents to execute.

        Returns:
            str: The final result after processing through all agents.

        Raises:
            ValueError: If task is None or empty
            Exception: If any error occurs during task execution
        """

        try:
            logger.info(
                f"Executing task with dynamic flow: {self.flow}"
            )
            return self.agent_rearrange.run(
                task,
                device=device,
                all_cpus=all_cpus,
                auto_gpu=auto_gpu,
                *args,
                **kwargs,
            )
        except Exception as e:
            logger.error(
                f"An error occurred while executing the task: {e}"
            )
            raise e

    def __call__(self, task: str, *args, **kwargs) -> str:
        return self.run(task, *args, **kwargs)

    def run_batched(self, tasks: List[str]) -> List[str]:
        """
        Executes a batch of tasks through the agents in the dynamically constructed flow.

        Args:
            tasks (List[str]): The tasks for the agents to execute.

        Returns:
            List[str]: The final results after processing through all agents.

        Raises:
            ValueError: If tasks is None or empty
            Exception: If any error occurs during task execution
        """
        if not tasks or not all(
            isinstance(task, str) for task in tasks
        ):
            raise ValueError(
                "Tasks must be a non-empty list of strings"
            )

        try:
            logger.info(
                f"Executing batch of tasks with dynamic flow: {self.flow}"
            )
            return [self.agent_rearrange.run(task) for task in tasks]
        except Exception as e:
            logger.error(
                f"An error occurred while executing the batch of tasks: {e}"
            )
            raise

    async def run_async(self, task: str) -> str:
        """
        Executes the task through the agents in the dynamically constructed flow asynchronously.

        Args:
            task (str): The task for the agents to execute.

        Returns:
            str: The final result after processing through all agents.

        Raises:
            ValueError: If task is None or empty
            Exception: If any error occurs during task execution
        """
        if not task or not isinstance(task, str):
            raise ValueError("Task must be a non-empty string")

        try:
            logger.info(
                f"Executing task with dynamic flow asynchronously: {self.flow}"
            )
            return await self.agent_rearrange.run_async(task)
        except Exception as e:
            logger.error(
                f"An error occurred while executing the task asynchronously: {e}"
            )
            raise

    async def run_concurrent(self, tasks: List[str]) -> List[str]:
        """
        Executes a batch of tasks through the agents in the dynamically constructed flow concurrently.

        Args:
            tasks (List[str]): The tasks for the agents to execute.

        Returns:
            List[str]: The final results after processing through all agents.

        Raises:
            ValueError: If tasks is None or empty
            Exception: If any error occurs during task execution
        """
        if not tasks or not all(
            isinstance(task, str) for task in tasks
        ):
            raise ValueError(
                "Tasks must be a non-empty list of strings"
            )

        try:
            logger.info(
                f"Executing batch of tasks with dynamic flow concurrently: {self.flow}"
            )
            with ThreadPoolExecutor() as executor:
                results = [
                    executor.submit(self.agent_rearrange.run, task)
                    for task in tasks
                ]
                return [
                    result.result()
                    for result in as_completed(results)
                ]
        except Exception as e:
            logger.error(
                f"An error occurred while executing the batch of tasks concurrently: {e}"
            )
            raise
