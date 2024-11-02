import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Union

from loguru import logger
from pydantic import BaseModel, Field

from swarms.structs.agent import Agent
from swarms.structs.concurrent_workflow import ConcurrentWorkflow
from swarms.structs.mixture_of_agents import MixtureOfAgents
from swarms.structs.rearrange import AgentRearrange
from swarms.structs.sequential_workflow import SequentialWorkflow
from swarms.structs.spreadsheet_swarm import SpreadSheetSwarm
from tenacity import retry, stop_after_attempt, wait_fixed
from swarms.structs.swarm_matcher import swarm_matcher
from swarms.prompts.ag_prompt import aggregator_system_prompt

SwarmType = Literal[
    "AgentRearrange",
    "MixtureOfAgents",
    "SpreadSheetSwarm",
    "SequentialWorkflow",
    "ConcurrentWorkflow",
    "auto",
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
    A class to dynamically route tasks to different swarm types based on user selection or automatic matching.

    This class enables users to specify a swarm type or let the system automatically determine the best swarm type for a given task. It then runs the task on the selected or matched swarm type, ensuring type validation, logging, and metadata capture.

    Attributes:
        name (str): The name of the SwarmRouter instance.
        description (str): A description of the SwarmRouter instance.
        max_loops (int): The maximum number of loops to perform.
        agents (List[Union[Agent, Callable]]): A list of Agent objects to be used in the swarm.
        swarm_type (SwarmType): The type of swarm to be used, which can be specified or automatically determined.
        autosave (bool): A flag to enable/disable autosave.
        flow (str): The flow of the swarm.
        return_json (bool): A flag to enable/disable returning the result in JSON format.
        auto_generate_prompts (bool): A flag to enable/disable auto generation of prompts.
        swarm (Union[AgentRearrange, MixtureOfAgents, SpreadSheetSwarm, SequentialWorkflow, ConcurrentWorkflow]):
            The instantiated swarm object.
        logs (List[SwarmLog]): A list of log entries captured during operations.
        auto_generate_prompt (bool): A flag to enable/disable auto generation of prompts.

    Available Swarm Types:
        - AgentRearrange: Rearranges agents for optimal task execution.
        - MixtureOfAgents: Combines different types of agents for diverse task handling.
        - SpreadSheetSwarm: Utilizes spreadsheet-like operations for task management.
        - SequentialWorkflow: Executes tasks in a sequential manner.
        - ConcurrentWorkflow: Executes tasks concurrently for parallel processing.
        - "auto" will automatically conduct embedding search to find the best swarm for your task
    """

    def __init__(
        self,
        name: str = "swarm-router",
        description: str = "Routes your task to the desired swarm",
        max_loops: int = 1,
        agents: List[Union[Agent, Callable]] = [],
        swarm_type: SwarmType = "SequentialWorkflow",  # "SpreadSheetSwarm" # "auto"
        autosave: bool = False,
        flow: str = None,
        return_json: bool = True,
        auto_generate_prompts: bool = False,
        *args,
        **kwargs,
    ):
        self.name = name
        self.description = description
        self.max_loops = max_loops
        self.agents = agents
        self.swarm_type = swarm_type
        self.autosave = autosave
        self.flow = flow
        self.return_json = return_json
        self.auto_generate_prompts = auto_generate_prompts
        self.logs = []

        self.reliability_check()

        self._log(
            "info",
            f"SwarmRouter initialized with swarm type: {swarm_type}",
        )

        self.activate_ape()

    def activate_ape(self):
        """Activate automatic prompt engineering for agents that support it"""
        try:
            logger.info("Activating automatic prompt engineering...")
            activated_count = 0
            for agent in self.agents:
                if hasattr(agent, "auto_generate_prompt"):
                    agent.auto_generate_prompt = (
                        self.auto_generate_prompts
                    )
                    activated_count += 1
                    logger.debug(
                        f"Activated APE for agent: {agent.name if hasattr(agent, 'name') else 'unnamed'}"
                    )

            logger.info(
                f"Successfully activated APE for {activated_count} agents"
            )
            self._log(
                "info",
                f"Activated automatic prompt engineering for {activated_count} agents",
            )

        except Exception as e:
            error_msg = f"Error activating automatic prompt engineering: {str(e)}"
            logger.error(error_msg)
            self._log("error", error_msg)
            raise RuntimeError(error_msg) from e

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def reliability_check(self):
        logger.info("Logger initializing checks")

        if not self.agents:
            raise ValueError("No agents provided for the swarm.")
        if self.swarm_type is None:
            raise ValueError("Swarm type cannot be 'none'.")
        if self.max_loops == 0:
            raise ValueError("max_loops cannot be 0.")

        logger.info("Checks completed your swarm is ready.")

    def _create_swarm(
        self, task: str = None, *args, **kwargs
    ) -> Union[
        AgentRearrange,
        MixtureOfAgents,
        SpreadSheetSwarm,
        SequentialWorkflow,
        ConcurrentWorkflow,
    ]:
        """
        Dynamically create and return the specified swarm type or automatically match the best swarm type for a given task.

        Args:
            task (str, optional): The task to be executed by the swarm. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Union[AgentRearrange, MixtureOfAgents, SpreadSheetSwarm, SequentialWorkflow, ConcurrentWorkflow]:
                The instantiated swarm object.

        Raises:
            ValueError: If an invalid swarm type is provided.
        """
        if self.swarm_type == "auto":
            self.swarm_type = str(swarm_matcher(task))

            self._create_swarm(self.swarm_type)

        elif self.swarm_type == "AgentRearrange":
            return AgentRearrange(
                name=self.name,
                description=self.description,
                agents=self.agents,
                max_loops=self.max_loops,
                flow=self.flow,
                return_json=self.return_json,
                *args,
                **kwargs,
            )
        elif self.swarm_type == "MixtureOfAgents":
            return MixtureOfAgents(
                name=self.name,
                description=self.description,
                reference_agents=self.agents,
                aggregator_system_prompt=aggregator_system_prompt.get_prompt(),
                aggregator_agent=self.agents[-1],
                layers=self.max_loops,
                *args,
                **kwargs,
            )
        elif self.swarm_type == "SpreadSheetSwarm":
            return SpreadSheetSwarm(
                name=self.name,
                description=self.description,
                agents=self.agents,
                max_loops=self.max_loops,
                autosave_on=self.autosave,
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
                auto_save=self.autosave,
                return_str_on=self.return_json,
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

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def run(self, task: str, *args, **kwargs) -> Any:
        """
        Dynamically run the specified task on the selected or matched swarm type.

        Args:
            task (str): The task to be executed by the swarm.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of the swarm's execution.

        Raises:
            Exception: If an error occurs during task execution.
        """
        self.swarm = self._create_swarm(task, *args, **kwargs)

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

    def batch_run(
        self, tasks: List[str], *args, **kwargs
    ) -> List[Any]:
        """
        Execute a batch of tasks on the selected or matched swarm type.

        Args:
            tasks (List[str]): A list of tasks to be executed by the swarm.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[Any]: A list of results from the swarm's execution.

        Raises:
            Exception: If an error occurs during task execution.
        """
        results = []
        for task in tasks:
            try:
                result = self.run(task, *args, **kwargs)
                results.append(result)
            except Exception as e:
                self._log(
                    "error",
                    f"Error occurred while running batch task on {self.swarm_type} swarm: {str(e)}",
                    task=task,
                    metadata={"error": str(e)},
                )
                raise
        return results

    def threaded_run(self, task: str, *args, **kwargs) -> Any:
        """
        Execute a task on the selected or matched swarm type using threading.

        Args:
            task (str): The task to be executed by the swarm.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of the swarm's execution.

        Raises:
            Exception: If an error occurs during task execution.
        """
        from threading import Thread

        def run_in_thread():
            try:
                result = self.run(task, *args, **kwargs)
                return result
            except Exception as e:
                self._log(
                    "error",
                    f"Error occurred while running task in thread on {self.swarm_type} swarm: {str(e)}",
                    task=task,
                    metadata={"error": str(e)},
                )
                raise

        thread = Thread(target=run_in_thread)
        thread.start()
        thread.join()
        return thread.result

    def async_run(self, task: str, *args, **kwargs) -> Any:
        """
        Execute a task on the selected or matched swarm type asynchronously.

        Args:
            task (str): The task to be executed by the swarm.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of the swarm's execution.

        Raises:
            Exception: If an error occurs during task execution.
        """
        import asyncio

        async def run_async():
            try:
                result = await asyncio.to_thread(
                    self.run, task, *args, **kwargs
                )
                return result
            except Exception as e:
                self._log(
                    "error",
                    f"Error occurred while running task asynchronously on {self.swarm_type} swarm: {str(e)}",
                    task=task,
                    metadata={"error": str(e)},
                )
                raise

        return asyncio.run(run_async())

    def get_logs(self) -> List[SwarmLog]:
        """
        Retrieve all logged entries.

        Returns:
            List[SwarmLog]: A list of all log entries.
        """
        return self.logs

    def concurrent_run(self, task: str, *args, **kwargs) -> Any:
        """
        Execute a task on the selected or matched swarm type concurrently.

        Args:
            task (str): The task to be executed by the swarm.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of the swarm's execution.

        Raises:
            Exception: If an error occurs during task execution.
        """
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor() as executor:
            future = executor.submit(self.run, task, *args, **kwargs)
            result = future.result()
            return result

    def concurrent_batch_run(
        self, tasks: List[str], *args, **kwargs
    ) -> List[Any]:
        """
        Execute a batch of tasks on the selected or matched swarm type concurrently.

        Args:
            tasks (List[str]): A list of tasks to be executed by the swarm.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[Any]: A list of results from the swarm's execution.

        Raises:
            Exception: If an error occurs during task execution.
        """
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.run, task, *args, **kwargs)
                for task in tasks
            ]
            results = [future.result() for future in futures]
            return results


def swarm_router(
    name: str = "swarm-router",
    description: str = "Routes your task to the desired swarm",
    max_loops: int = 1,
    agents: List[Union[Agent, Callable]] = [],
    swarm_type: SwarmType = "SequentialWorkflow",  # "SpreadSheetSwarm" # "auto"
    autosave: bool = False,
    flow: str = None,
    return_json: bool = True,
    auto_generate_prompts: bool = False,
    task: str = None,
    *args,
    **kwargs,
) -> SwarmRouter:
    """
    Create and run a SwarmRouter instance with the given configuration.

    Args:
        name (str, optional): Name of the swarm router. Defaults to "swarm-router".
        description (str, optional): Description of the router. Defaults to "Routes your task to the desired swarm".
        max_loops (int, optional): Maximum number of execution loops. Defaults to 1.
        agents (List[Union[Agent, Callable]], optional): List of agents or callables. Defaults to [].
        swarm_type (SwarmType, optional): Type of swarm to use. Defaults to "SequentialWorkflow".
        autosave (bool, optional): Whether to autosave results. Defaults to False.
        flow (str, optional): Flow configuration. Defaults to None.
        return_json (bool, optional): Whether to return results as JSON. Defaults to True.
        auto_generate_prompts (bool, optional): Whether to auto-generate prompts. Defaults to False.
        task (str, optional): Task to execute. Defaults to None.
        *args: Additional positional arguments passed to SwarmRouter.run()
        **kwargs: Additional keyword arguments passed to SwarmRouter.run()

    Returns:
        Any: Result from executing the swarm router

    Raises:
        ValueError: If invalid arguments are provided
        Exception: If an error occurs during router creation or task execution
    """
    try:
        logger.info(
            f"Creating SwarmRouter with name: {name}, swarm_type: {swarm_type}"
        )

        if not agents:
            logger.warning(
                "No agents provided, router may have limited functionality"
            )

        if task is None:
            logger.warning("No task provided")

        swarm_router = SwarmRouter(
            name=name,
            description=description,
            max_loops=max_loops,
            agents=agents,
            swarm_type=swarm_type,
            autosave=autosave,
            flow=flow,
            return_json=return_json,
            auto_generate_prompts=auto_generate_prompts,
        )

        logger.info(f"Executing task with SwarmRouter: {task}")
        result = swarm_router.run(task, *args, **kwargs)
        logger.info("Task execution completed successfully")
        return result

    except ValueError as e:
        logger.error(
            f"Invalid arguments provided to swarm_router: {str(e)}"
        )
        raise
    except Exception as e:
        logger.error(f"Error in swarm_router execution: {str(e)}")
        raise
