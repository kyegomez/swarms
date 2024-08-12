import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from swarms.structs.agent import Agent
from swarms.structs.task import Task
from swarms.utils.logger import logger
from swarms.structs.base_swarm import BaseSwarm


@dataclass
class AsyncWorkflow(BaseSwarm):
    """
    Represents an asynchronous workflow to run tasks.

    Attributes:
        name (str): The name of the workflow.
        description (str): The description of the workflow.
        max_loops (int): The maximum number of loops to run the workflow.
        autosave (bool): Flag indicating whether to autosave the results.
        dashboard (bool): Flag indicating whether to display a dashboard.
        task_pool (List[Any]): The list of tasks in the workflow.
        results (List[Any]): The list of results from running the tasks.
        loop (Optional[asyncio.AbstractEventLoop]): The event loop to use.
        stopping_condition (Optional[Callable]): The stopping condition for the workflow.

    Methods:
        add(tasks: List[Any]) -> None:
            Add tasks to the workflow.

        delete(task: Task = None, tasks: List[Task] = None) -> None:
            Delete a task from the workflow.

        run() -> List[Any]:
            Run the workflow and return the results.
    """

    name: str = "Async Workflow"
    description: str = "A workflow to run asynchronous tasks"
    max_loops: int = 1
    autosave: bool = True
    dashboard: bool = False
    task_pool: List[Any] = field(default_factory=list)
    results: List[Any] = field(default_factory=list)
    loop: Optional[asyncio.AbstractEventLoop] = None
    stopping_condition: Optional[Callable] = None
    agents: List[Agent] = None

    async def add(self, task: Any = None, tasks: List[Any] = None):
        """Add tasks to the workflow"""
        try:
            if tasks:
                for task in tasks:
                    self.task_pool.extend(tasks)
            elif task:
                self.task_pool.append(task)

            else:
                if task and tasks:
                    # Add the task and tasks to the task pool
                    self.task_pool.append(task)
                    self.task_pool.extend(tasks)
                else:
                    raise ValueError(
                        "Either task or tasks must be provided"
                    )

        except Exception as error:
            logger.error(f"[ERROR][AsyncWorkflow] {error}")

    async def delete(self, task: Any = None, tasks: List[Task] = None):
        """Delete a task from the workflow"""
        try:
            if task:
                self.task_pool.remove(task)
            elif tasks:
                for task in tasks:
                    self.task_pool.remove(task)
        except Exception as error:
            logger.error(f"[ERROR][AsyncWorkflow] {error}")

    async def run(self):
        """Run the workflow"""
        if self.loop is None:
            self.loop = asyncio.get_event_loop()
        for i in range(self.max_loops):
            logger.info(
                f"[INFO][AsyncWorkflow] Loop {i + 1}/{self.max_loops}"
            )
            futures = [
                asyncio.ensure_future(task.execute())
                for task in self.task_pool
            ]
            self.results = await asyncio.gather(*futures)
            # if self.autosave:
            #     self.save()
            # if self.dashboard:
            #     self.display()

            # Add a stopping condition to stop the workflow, if provided but stopping_condition takes in a parameter s for string
            if self.stopping_condition:
                if self.stopping_condition(self.results):
                    break

        return self.results
