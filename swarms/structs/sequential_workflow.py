import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Union

from swarms.prompts.multi_agent_collab_prompt import (
    MULTI_AGENT_COLLAB_PROMPT,
)
from swarms.structs.agent import Agent
from swarms.structs.agent_rearrange import AgentRearrange
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType

logger = initialize_logger(log_folder="sequential_workflow")


class SequentialWorkflow:
    """
    Orchestrates the execution of a sequence of agents in a defined workflow.

    This class enables the construction and execution of a workflow where multiple agents
    (or callables) are executed in a specified order, passing tasks and optional data
    through the chain. It supports both synchronous and asynchronous execution, as well as
    batched and concurrent task processing.

    Attributes:
        id (str): Unique identifier for the workflow instance.
        name (str): Human-readable name for the workflow.
        description (str): Description of the workflow's purpose.
        agents (List[Union[Agent, Callable]]): List of agents or callables to execute in sequence.
        max_loops (int): Maximum number of times to execute the workflow.
        output_type (OutputType): Format of the output from the workflow.
        shared_memory_system (callable): Optional callable for managing shared memory between agents.
        multi_agent_collab_prompt (bool): Whether to append a collaborative prompt to each agent.
        flow (str): String representation of the agent execution order.
        agent_rearrange (AgentRearrange): Internal helper for managing agent execution.

    Raises:
        ValueError: If the agents list is None or empty, or if max_loops is set to 0.
    """

    def __init__(
        self,
        id: str = "sequential_workflow",
        name: str = "SequentialWorkflow",
        description: str = "Sequential Workflow, where agents are executed in a sequence.",
        agents: List[Union[Agent, Callable]] = None,
        max_loops: int = 1,
        output_type: OutputType = "dict",
        shared_memory_system: callable = None,
        multi_agent_collab_prompt: bool = False,
        team_awareness: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize a SequentialWorkflow instance.

        Args:
            id (str, optional): Unique identifier for the workflow. Defaults to "sequential_workflow".
            name (str, optional): Name of the workflow. Defaults to "SequentialWorkflow".
            description (str, optional): Description of the workflow. Defaults to a standard description.
            agents (List[Union[Agent, Callable]], optional): List of agents or callables to execute in sequence.
            max_loops (int, optional): Maximum number of times to execute the workflow. Defaults to 1.
            output_type (OutputType, optional): Output format for the workflow. Defaults to "dict".
            shared_memory_system (callable, optional): Callable for shared memory management. Defaults to None.
            multi_agent_collab_prompt (bool, optional): If True, appends a collaborative prompt to each agent.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If the agents list is None or empty, or if max_loops is set to 0.
        """
        self.id = id
        self.name = name
        self.description = description
        self.agents = agents
        self.max_loops = max_loops
        self.output_type = output_type
        self.shared_memory_system = shared_memory_system
        self.multi_agent_collab_prompt = multi_agent_collab_prompt
        self.team_awareness = team_awareness

        self.reliability_check()
        self.flow = self.sequential_flow()

        self.agent_rearrange = AgentRearrange(
            name=self.name,
            description=self.description,
            agents=self.agents,
            flow=self.flow,
            max_loops=self.max_loops,
            output_type=self.output_type,
            team_awareness=self.team_awareness,
        )

    def reliability_check(self):
        """
        Validates the workflow configuration and prepares agents for execution.

        Raises:
            ValueError: If the agents list is None or empty, or if max_loops is set to 0.
        """
        if self.agents is None or len(self.agents) == 0:
            raise ValueError("Agents list cannot be None or empty")

        if self.max_loops == 0:
            raise ValueError("max_loops cannot be 0")

        if self.multi_agent_collab_prompt is True:
            for agent in self.agents:
                if hasattr(agent, "system_prompt"):
                    if agent.system_prompt is None:
                        agent.system_prompt = ""
                    agent.system_prompt += MULTI_AGENT_COLLAB_PROMPT
                else:
                    logger.warning(
                        f"Agent {getattr(agent, 'name', str(agent))} does not have a 'system_prompt' attribute."
                    )

        logger.info(
            f"Sequential Workflow Name: {self.name} is ready to run."
        )

    def sequential_flow(self):
        """
        Constructs a string representation of the agent execution order.

        Returns:
            str: A string showing the order of agent execution (e.g., "AgentA -> AgentB -> AgentC").
                 Returns an empty string if no valid agent names are found.
        """
        if self.agents:
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

    def run(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
        *args,
        **kwargs,
    ):
        """
        Executes a specified task through the agents in the dynamically constructed flow.

        Args:
            task (str): The task for the agents to execute.
            img (Optional[str], optional): An optional image input for the agents.
            imgs (Optional[List[str]], optional): Optional list of images for the agents.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The final result after processing through all agents.

        Raises:
            ValueError: If the task is None or empty.
            Exception: If any error occurs during task execution.
        """
        try:
            # prompt = f"{MULTI_AGENT_COLLAB_PROMPT}\n\n{task}"
            return self.agent_rearrange.run(
                task=task,
                img=img,
                streaming_callback=streaming_callback,
                *args,
                **kwargs,
            )

        except Exception as e:
            logger.error(
                f"An error occurred while executing the task: {e}"
            )
            raise e

    def __call__(self, task: str, *args, **kwargs):
        """
        Allows the SequentialWorkflow instance to be called as a function.

        Args:
            task (str): The task for the agents to execute.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The final result after processing through all agents.
        """
        return self.run(task, *args, **kwargs)

    def run_batched(self, tasks: List[str]) -> List[str]:
        """
        Executes a batch of tasks through the agents in the dynamically constructed flow.

        Args:
            tasks (List[str]): A list of tasks for the agents to execute.

        Returns:
            List[str]: A list of final results after processing through all agents.

        Raises:
            ValueError: If tasks is None, empty, or contains non-string elements.
            Exception: If any error occurs during task execution.
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
        Executes the specified task through the agents in the dynamically constructed flow asynchronously.

        Args:
            task (str): The task for the agents to execute.

        Returns:
            str: The final result after processing through all agents.

        Raises:
            ValueError: If task is None or not a string.
            Exception: If any error occurs during task execution.
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
            tasks (List[str]): A list of tasks for the agents to execute.

        Returns:
            List[str]: A list of final results after processing through all agents.

        Raises:
            ValueError: If tasks is None, empty, or contains non-string elements.
            Exception: If any error occurs during task execution.
        """
        if not tasks or not all(
            isinstance(task, str) for task in tasks
        ):
            raise ValueError(
                "Tasks must be a non-empty list of strings"
            )

        try:
            with ThreadPoolExecutor(
                max_workers=os.cpu_count()
            ) as executor:
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
