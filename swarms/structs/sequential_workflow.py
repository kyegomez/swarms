from typing import List, Optional
from swarms.structs.agent import Agent
from swarms.structs.rearrange import AgentRearrange
from swarms.structs.output_types import OutputType
from concurrent.futures import ThreadPoolExecutor, as_completed
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="sequential_workflow")


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
        self.flow = self.sequential_flow()

        self.agent_rearrange = AgentRearrange(
            name=name,
            description=description,
            agents=agents,
            flow=self.flow,
            max_loops=max_loops,
            output_type=output_type,
            return_json=return_json,
            shared_memory_system=shared_memory_system,
            *args,
            **kwargs,
        )

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

        logger.info("Checks completed your swarm is ready.")

    def run(
        self,
        task: str,
        img: Optional[str] = None,
        device: str = "cpu",
        all_cores: bool = False,
        all_gpus: bool = False,
        device_id: int = 0,
        no_use_clusterops: bool = True,
        *args,
        **kwargs,
    ) -> str:
        """
        Executes a task through the agents in the dynamically constructed flow.

        Args:
            task (str): The task for the agents to execute.
            device (str): The device to use for the agents to execute.
            all_cores (bool): Whether to use all cores.
            all_gpus (bool): Whether to use all gpus.
            device_id (int): The device id to use for the agents to execute.
            no_use_clusterops (bool): Whether to use clusterops.


        Returns:
            str: The final result after processing through all agents.

        Raises:
            ValueError: If task is None or empty
            Exception: If any error occurs during task execution
        """

        try:
            return self.agent_rearrange.run(
                task=task,
                img=img,
                device=device,
                all_cores=all_cores,
                device_id=device_id,
                all_gpus=all_gpus,
                no_use_clusterops=no_use_clusterops,
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
