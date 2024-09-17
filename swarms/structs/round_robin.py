import random
from swarms.structs.base_swarm import BaseSwarm
from typing import List
from swarms.structs.agent import Agent
from swarms.utils.loguru_logger import logger
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from swarms.schemas.agent_step_schemas import ManySteps

datetime_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class MetadataSchema(BaseModel):
    swarm_id: Optional[str] = Field(
        ..., description="Unique ID for the run"
    )
    name: Optional[str] = Field(
        "RoundRobinSwarm", description="Name of the swarm"
    )
    task: Optional[str] = Field(
        ..., description="Task or query given to all agents"
    )
    description: Optional[str] = Field(
        "Concurrent execution of multiple agents",
        description="Description of the workflow",
    )
    agent_outputs: Optional[List[ManySteps]] = Field(
        ..., description="List of agent outputs and metadata"
    )
    timestamp: Optional[str] = Field(
        default_factory=datetime.now,
        description="Timestamp of the workflow execution",
    )
    max_loops: Optional[int] = Field(
        1, description="Maximum number of loops to run"
    )


class RoundRobinSwarm(BaseSwarm):
    """
    A swarm implementation that executes tasks in a round-robin fashion.

    Args:
        agents (List[Agent], optional): List of agents in the swarm. Defaults to None.
        verbose (bool, optional): Flag to enable verbose mode. Defaults to False.
        max_loops (int, optional): Maximum number of loops to run. Defaults to 1.
        callback (callable, optional): Callback function to be called after each loop. Defaults to None.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        agents (List[Agent]): List of agents in the swarm.
        verbose (bool): Flag to enable verbose mode.
        max_loops (int): Maximum number of loops to run.
        index (int): Current index of the agent being executed.

    Methods:
        run(task: str, *args, **kwargs) -> Any: Executes the given task on the agents in a round-robin fashion.

    """

    def __init__(
        self,
        name: str = "RoundRobinSwarm",
        description: str = "A swarm implementation that executes tasks in a round-robin fashion.",
        agents: List[Agent] = None,
        verbose: bool = False,
        max_loops: int = 1,
        callback: callable = None,
        return_json_on: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            agents=agents,
            *args,
            **kwargs,
        )
        self.name = name
        self.description = description
        self.agents = agents
        self.verbose = verbose
        self.max_loops = max_loops
        self.callback = callback
        self.return_json_on = return_json_on
        self.index = 0

        # Store the metadata for the run
        self.output_schema = MetadataSchema(
            name=self.name,
            swarm_id=datetime_stamp,
            task="",
            description=self.description,
            agent_outputs=[],
            timestamp=datetime_stamp,
            max_loops=self.max_loops,
        )

        # Set the max loops for every agent
        for agent in self.agents:
            agent.max_loops = random.randint(1, 5)

    def run(self, task: str, *args, **kwargs):
        """
        Executes the given task on the agents in a round-robin fashion.

        Args:
            task (str): The task to be executed.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of the task execution.

        Raises:
            Exception: If an exception occurs during task execution.

        """
        try:
            result = task
            self.output_schema.task = task
            n = len(self.agents)
            logger.info(f"Running the task {task} on {n} agents.")
            for loop in range(self.max_loops):
                for _ in range(n):
                    current_agent = self.agents[self.index]
                    try:
                        logger.info(
                            f"Running Agent {current_agent.agent_name} on task {result}"
                        )
                        result = current_agent.run(
                            result, *args, **kwargs
                        )

                        # Add agent schema to output
                        self.output_schema.agent_outputs.append(
                            current_agent.agent_output
                        )
                    except Exception as e:
                        logger.error(
                            f"Handling an exception for {current_agent.name}: {e}"
                        )
                        raise e
                    finally:
                        self.index = (
                            self.index + 1
                        ) % n  # Increment and wrap around the index
                if self.callback:
                    logger.info(
                        f"Calling the callback function for loop {loop}"
                    )
                    self.callback(loop, result)

            if self.return_json_on:
                return self.export_metadata()
            else:
                return result
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return e

    def export_metadata(self):
        return self.output_schema.model_dump_json(indent=4)
