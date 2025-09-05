import traceback
from typing import List

from loguru import logger

from swarms.structs.multi_agent_exec import (
    batched_grid_agent_execution,
)
from swarms.structs.omni_agent_types import AgentType
from swarms.structs.swarm_id import swarm_id


class BatchedGridWorkflow:
    def __init__(
        self,
        id: str = swarm_id(),
        name: str = "BatchedGridWorkflow",
        description: str = "For every agent, run the task on a different task",
        agents: List[AgentType] = None,
        max_loops: int = 1,
    ):
        """
        Initialize a BatchedGridWorkflow instance.

        Args:
            id: Unique identifier for the workflow
            name: Name of the workflow
            description: Description of what the workflow does
            agents: List of agents to execute tasks
            max_loops: Maximum number of execution loops to run (must be >= 1)
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
            tasks: List of tasks to execute

        Returns:
            Output from the batched grid agent execution
        """
        return batched_grid_agent_execution(self.agents, tasks)

    def run_(self, tasks: List[str]):
        """
        Run the batched grid workflow with the given tasks.

        Args:
            tasks: List of tasks to execute

        Returns:
            List: Results from all execution loops
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
        """
        try:
            return self.run_(tasks)
        except Exception as e:
            logger.error(
                f"BatchedGridWorkflow Error: {self.name}\n\nId: {self.id}\n\nAn error occurred while running the batched grid workflow: {e}\nTraceback:\n{traceback.format_exc()}"
            )
            raise e
