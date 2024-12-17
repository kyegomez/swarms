import asyncio
from typing import Any, List
from swarms.structs.base_workflow import BaseWorkflow
from swarms.structs.agent import Agent
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger("async_workflow")

class AsyncWorkflow(BaseWorkflow):
    """
    Represents an asynchronous workflow that can execute tasks concurrently using multiple agents.
    
    Attributes:
    - name (str): The name of the workflow.
    - agents (List[Agent]): A list of agents participating in the workflow.
    - max_workers (int): The maximum number of workers to use for concurrent execution.
    - dashboard (bool): Indicates if a dashboard should be displayed.
    - autosave (bool): Indicates if the results should be autosaved.
    - verbose (bool): Indicates if verbose logging is enabled.
    - task_pool (List): A pool of tasks to be executed.
    - results (List): The results of the executed tasks.
    - loop (asyncio.AbstractEventLoop): The event loop used for asynchronous execution.
    """
    def __init__(
        self,
        name: str = "AsyncWorkflow",
        agents: List[Agent] = None,
        max_workers: int = 5,
        dashboard: bool = False,
        autosave: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(agents=agents, **kwargs)
        self.name = name
        self.agents = agents or []
        self.max_workers = max_workers
        self.dashboard = dashboard
        self.autosave = autosave
        self.verbose = verbose
        self.task_pool = []
        self.results = []
        self.loop = None

    async def _execute_agent_task(
        self, agent: Agent, task: str
    ) -> Any:
        """
        Executes a single agent task asynchronously.
        
        Args:
        - agent (Agent): The agent executing the task.
        - task (str): The task to be executed.
        
        Returns:
        - Any: The result of the task execution or an error message if an exception occurs.
        """
        try:
            if self.verbose:
                logger.info(
                    f"Agent {agent.agent_name} processing task: {task}"
                )
            result = await agent.arun(task)
            if self.verbose:
                logger.info(
                    f"Agent {agent.agent_name} completed task"
                )
            return result
        except Exception as e:
            logger.error(
                f"Error in agent {agent.agent_name}: {str(e)}"
            )
            return str(e)

    async def run(self, task: str) -> List[Any]:
        """
        Runs the workflow with all agents processing the task concurrently.
        
        Args:
        - task (str): The task to be executed by all agents.
        
        Returns:
        - List[Any]: A list of results from all agents or error messages if exceptions occur.
        """
        if not self.agents:
            raise ValueError("No agents provided to the workflow")

        try:
            # Create tasks for all agents
            tasks = [
                self._execute_agent_task(agent, task)
                for agent in self.agents
            ]

            # Execute all tasks concurrently
            self.results = await asyncio.gather(
                *tasks, return_exceptions=True
            )

            if self.autosave:
                # TODO: Implement autosave logic here
                pass

            return self.results

        except Exception as e:
            logger.error(f"Error in workflow execution: {str(e)}")
            raise
