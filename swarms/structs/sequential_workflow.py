import time
import json

from swarms.utils.loguru_logger import logger
from swarms.structs.base_workflow import BaseWorkflow
from pydantic import BaseModel, Field
from typing import List, Dict
from swarms.structs.agent import Agent


class StepSequentialWorkflow(BaseModel):
    agent_names: List[str] = Field(
        ..., description="List of agent names to include in the workflow."
    )
    max_loops: int = Field(
        1, description="Maximum number of loops to run the workflow."
    )
    verbose: bool = Field(
        False, description="Whether to log debug information."
    )
    steps: Dict = Field(
        ...,
        description="Dictionary of steps for the workflow with each agent and its parameters.",
    )
    time: str = Field(
        time.strftime("%Y-%m-%d %H:%M:%S"),
        description="Time of the workflow.",
    )


# Define a class to handle the sequential workflow
class SequentialWorkflow(BaseWorkflow):
    def __init__(
        self,
        agents: List[Agent] = None,
        max_loops: int = 2,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initializes a SequentialWorkflow with a list of agents.

        :param agents: List of agents to include in the workflow.
        """
        self.agents = agents
        self.max_loops = max_loops

        if verbose:
            logger.add("sequential_workflow.log", level="DEBUG")

        if not self.agents:
            raise ValueError("No agents provided for workflow")

        if not self.max_loops:
            self.max_loops = 1

        # Log all the agents in the workflow
        logger.info(
            f"Initialized SequentialWorkflow with agents: {json.dumps([str(agent.agent_name) for agent in self.agents])}"
        )

    def run(self, task: str, *args, **kwargs):
        """
        Run the workflow starting with an initial task.

        :param task: The task to start the workflow.
        """
        logger.info(f"Starting workflow with task: {task}")
        current_output = task
        for agent in self.agents:
            count = 0
            while count < self.max_loops:
                try:
                    logger.info(f"Running agent {agent.agent_name}")
                    current_output = agent.run(
                        current_output, *args, **kwargs
                    )
                    print(current_output)
                    count += 1
                    logger.debug(
                        f"Agent {agent.agent_name} completed loop {count} "
                    )  # Log partial output for brevity
                except Exception as e:
                    logger.error(
                        f"Error occurred while running agent {agent.agent_name}: {str(e)}"
                    )
                    raise
            logger.info(f"Finished running agent {agent.agent_name}")
        logger.info("Finished running workflow")
        return current_output
