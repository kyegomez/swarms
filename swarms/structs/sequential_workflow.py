from typing import List
from swarms import Agent
from swarms.utils.loguru_logger import logger
from swarms.structs.rearrange import AgentRearrange


class SequentialWorkflow:
    """
    Initializes a SequentialWorkflow object.

    Args:
        agents (List[Agent], optional): The list of agents in the workflow. Defaults to None.
        max_loops (int, optional): The maximum number of loops to execute the workflow. Defaults to 1.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
        self,
        agents: List[Agent] = None,
        max_loops: int = 1,
        *args,
        **kwargs,
    ):
        self.agents = agents
        self.flow = " -> ".join(agent.agent_name for agent in agents)
        self.agent_rearrange = AgentRearrange(
            agents, self.flow, max_loops=max_loops, *args, **kwargs
        )

    def run(self, task: str) -> str:
        """
        Runs the task through the agents in the dynamically constructed flow.

        Args:
            task (str): The task for the agents to execute.

        Returns:
            str: The final result after processing through all agents.
        """
        try:
            logger.info(f"Running task with dynamic flow: {self.flow}")
            return self.agent_rearrange.run(task)
        except Exception as e:
            logger.error(f"An error occurred while running the task: {e}")
            raise
