import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Union

from loguru import logger
from pydantic import BaseModel, Field
from swarms.structs.agent import Agent
from swarms.structs.concurrent_workflow import ConcurrentWorkflow
from swarms.structs.mixture_of_agents import MixtureOfAgents
from swarms.structs.rearrange import AgentRearrange
from swarms.structs.sequential_workflow import SequentialWorkflow
from swarms.structs.spreadsheet_swarm import SpreadSheetSwarm

SwarmType = Literal[
    "AgentRearrange",
    "MixtureOfAgents",
    "SpreadSheetSwarm",
    "SequentialWorkflow",
    "ConcurrentWorkflow",
]


class SwarmLog(BaseModel):
    """
    A Pydantic model to capture log entries.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    level: str
    message: str
    swarm_type: SwarmType
    task: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SwarmRouter:
    """
    A class to route tasks to different swarm types based on user selection.

    This class allows users to specify a swarm type and a list of agents, then run tasks
    on the selected swarm type. It includes type validation, logging, and metadata capture.

    Attributes:
        agents (List[Agent]): A list of Agent objects to be used in the swarm.
        swarm_type (SwarmType): The type of swarm to be used.
        swarm (Union[AgentRearrange, GraphWorkflow, MixtureOfAgents, SpreadSheetSwarm]):
            The instantiated swarm object.
        logs (List[SwarmLog]): A list of log entries captured during operations.

    Available Swarm Types:
        - AgentRearrange: Rearranges agents for optimal task execution.
        - MixtureOfAgents: Combines different types of agents for diverse task handling.
        - SpreadSheetSwarm: Utilizes spreadsheet-like operations for task management.
        - SequentialWorkflow: Executes tasks in a sequential manner.
        - ConcurrentWorkflow: Executes tasks concurrently for parallel processing.
    """

    def __init__(
        self,
        name: str = "swarm-router",
        description: str = "Routes your task to the desired swarm",
        max_loops: int = 1,
        agents: List[Agent] = None,
        swarm_type: SwarmType = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the SwarmRouter with a list of agents and a swarm type.

        Args:
            name (str, optional): The name of the SwarmRouter instance. Defaults to None.
            description (str, optional): A description of the SwarmRouter instance. Defaults to None.
            max_loops (int, optional): The maximum number of loops to perform. Defaults to 1.
            agents (List[Agent], optional): A list of Agent objects to be used in the swarm. Defaults to None.
            swarm_type (SwarmType, optional): The type of swarm to be used. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            ValueError: If an invalid swarm type is provided, or if there are no agents, or if swarm type is "none", or if max_loops is 0.
        """
        if not agents:
            raise ValueError("No agents provided for the swarm.")
        if swarm_type is None:
            raise ValueError("Swarm type cannot be 'none'.")
        if max_loops == 0:
            raise ValueError("max_loops cannot be 0.")

        self.name = name
        self.description = description
        self.max_loops = max_loops
        self.agents = agents
        self.swarm_type = swarm_type
        self.swarm = self._create_swarm(*args, **kwargs)
        self.logs = []

        self._log(
            "info",
            f"SwarmRouter initialized with swarm type: {swarm_type}",
        )

    def _create_swarm(self, *args, **kwargs) -> Union[
        AgentRearrange,
        MixtureOfAgents,
        SpreadSheetSwarm,
    ]:
        """
        Create and return the specified swarm type.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Union[AgentRearrange, GraphWorkflow, MixtureOfAgents, SpreadSheetSwarm]:
                The instantiated swarm object.

        Raises:
            ValueError: If an invalid swarm type is provided.
        """
        if self.swarm_type == "AgentRearrange":
            return AgentRearrange(
                name=self.name,
                description=self.description,
                agents=self.agents,
                max_loops=self.max_loops,
                *args,
                **kwargs,
            )
        elif self.swarm_type == "MixtureOfAgents":
            return MixtureOfAgents(
                name=self.name,
                description=self.description,
                agents=self.agents,
                aggregator_agent=[self.agents[-1]],
                layers=self.max_loops,
                *args,
                **kwargs,
            )
        elif self.swarm_type == "SpreadSheetSwarm":
            return SpreadSheetSwarm(
                name=self.name,
                description=self.description,
                agents=self.agents,
                max_loops=1,
                *args,
                **kwargs,
            )
        elif self.swarm_type == "SequentialWorkflow":
            return SequentialWorkflow(
                name=self.name,
                description=self.description,
                agents=self.agents,
                max_loops=self.max_loops,
                *args,
                **kwargs,
            )
        elif self.swarm_type == "ConcurrentWorkflow":
            return ConcurrentWorkflow(
                name=self.name,
                description=self.description,
                agents=self.agents,
                max_loops=self.max_loops,
                *args,
                **kwargs,
            )
        else:
            raise ValueError(f"Invalid swarm type: {self.swarm_type}")

    def _log(
        self,
        level: str,
        message: str,
        task: str = "",
        metadata: Dict[str, Any] = None,
    ):
        """
        Create a log entry and add it to the logs list.

        Args:
            level (str): The log level (e.g., "info", "error").
            message (str): The log message.
            task (str, optional): The task being performed. Defaults to "".
            metadata (Dict[str, Any], optional): Additional metadata. Defaults to None.
        """
        log_entry = SwarmLog(
            level=level,
            message=message,
            swarm_type=self.swarm_type,
            task=task,
            metadata=metadata or {},
        )
        self.logs.append(log_entry)
        logger.log(level.upper(), message)

    def run(self, task: str, *args, **kwargs) -> Any:
        """
        Run the specified task on the selected swarm.

        Args:
            task (str): The task to be executed by the swarm.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of the swarm's execution.

        Raises:
            Exception: If an error occurs during task execution.
        """
        try:
            self._log(
                "info",
                f"Running task on {self.swarm_type} swarm",
                task=task,
                metadata=kwargs,
            )
            result = self.swarm.run(task, *args, **kwargs)
            self._log(
                "success",
                f"Task completed successfully on {self.swarm_type} swarm",
                task=task,
                metadata={"result": str(result)},
            )
            return result
        except Exception as e:
            self._log(
                "error",
                f"Error occurred while running task on {self.swarm_type} swarm: {str(e)}",
                task=task,
                metadata={"error": str(e)},
            )
            raise

    def get_logs(self) -> List[SwarmLog]:
        """
        Retrieve all logged entries.

        Returns:
            List[SwarmLog]: A list of all log entries.
        """
        return self.logs
