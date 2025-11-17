import traceback
from typing import List

from loguru import logger

from swarms.structs.multi_agent_exec import (
    batched_grid_agent_execution,
)
from swarms.structs.omni_agent_types import AgentType
from swarms.structs.swarm_id import swarm_id
from swarms.utils.output_types import OutputType


class BatchedGridWorkflow:
    """
    A workflow class for executing tasks in a batched grid pattern.

    This class implements a batched grid workflow where multiple agents can execute
    tasks concurrently in a grid-like fashion. Each agent processes tasks independently,
    and the workflow can be run for multiple loops to enable iterative processing.

    The workflow supports:
    - Concurrent task execution across multiple agents
    - Configurable number of execution loops
    - Error handling and logging for robust operation
    - Unique identification and naming for workflow instances

    Attributes:
        id (str): Unique identifier for the workflow instance.
        name (str): Human-readable name for the workflow.
        description (str): Description of the workflow's purpose.
        agents (List[AgentType]): List of agents to execute tasks.
        max_loops (int): Maximum number of execution loops to perform.
        output_type (OutputType): Type of output to return.

    Example:
        >>> from swarms.structs.batched_grid_workflow import BatchedGridWorkflow
        >>> workflow = BatchedGridWorkflow(
        ...     name="Data Processing Workflow",
        ...     agents=[agent1, agent2, agent3],
        ...     max_loops=3
        ... )
        >>> results = workflow.run(["task1", "task2", "task3"])
    """

    def __init__(
        self,
        id: str = swarm_id(),
        name: str = "BatchedGridWorkflow",
        description: str = "For every agent, run the task on a different task",
        agents: List[AgentType] = None,
        max_loops: int = 1,
        output_type: OutputType = "dict",
    ):
        """
        Initialize a BatchedGridWorkflow instance.

        Args:
            id: Unique identifier for the workflow.
            name: Name of the workflow.
            description: Description of what the workflow does.
            agents: List of agents to execute tasks.
            max_loops: Maximum number of execution loops to run (must be >= 1).
        """
        self.id = id
        self.name = name
        self.description = description
        self.agents = agents
        self.max_loops = max_loops

        # Validate max_loops parameter
        if not isinstance(max_loops, int) or max_loops < 1:
            raise ValueError("max_loops must be a positive integer")

    def step(self, tasks: List[str]):
        """
        Execute one step of the batched grid workflow.

        Args:
            tasks: List of tasks to execute.

        Returns:
            Output from the batched grid agent execution.
        """
        return batched_grid_agent_execution(self.agents, tasks)

    def run_(self, tasks: List[str]):
        """
        Run the batched grid workflow with the given tasks.

        Args:
            tasks: List of tasks to execute.

        Returns:
            List: Results from all execution loops.
        """
        results = []
        current_loop = 0

        while current_loop < self.max_loops:
            # Run the step with the original tasks
            output = self.step(tasks)
            results.append(output)
            current_loop += 1

        return results

    def run(self, tasks: List[str]):
        """
        Run the batched grid workflow with the given tasks.

        Args:
            tasks: List of tasks to execute.

        Returns:
            List: Results from all execution loops.

        Raises:
            Exception: If an error occurs during workflow execution.
        """
        try:
            return self.run_(tasks)
        except Exception as e:
            logger.error(
                (
                    f"BatchedGridWorkflow Error: {self.name}\n"
                    f"Id: {self.id}\n"
                    f"An error occurred while running the batched grid workflow: {e}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
            )
            raise
