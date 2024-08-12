import logging
from typing import List

from swarms.structs.base_structure import BaseStructure
from swarms.structs.task import Task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecursiveWorkflow(BaseStructure):
    """
    RecursiveWorkflow class for running a task recursively until a stopping condition is met.

    Args:
        task (Task): The task to execute.
        stop_token (Any): The token that indicates when to stop the workflow.

    Attributes:
        task (Task): The task to execute.
        stop_token (Any): The token that indicates when to stop the workflow.

    Examples:
    >>> from swarms.models import OpenAIChat
    >>> from swarms.structs import RecursiveWorkflow, Task
    >>> llm = OpenAIChat(openai_api_key="")
    >>> task = Task(llm, "What's the weather in miami")
    >>> workflow = RecursiveWorkflow()
    >>> workflow.add(task)
    >>> workflow.run()
    """

    def __init__(
        self,
        stop_token: str = "<DONE>",
        stopping_conditions: callable = None,
        max_loops: int = 1,
        *args,
        **kwargs,
    ):
        self.stop_token = stop_token
        self.stopping_conditions = stopping_conditions
        self.task_pool = []

        assert self.stop_token is not None, "stop_token cannot be None"

    def add(self, task: Task = None, tasks: List[Task] = None):
        """Adds a task to the workflow.

        Args:
            task (Task): _description_
            tasks (List[Task]): _description_
        """
        try:
            if tasks:
                for task in tasks:
                    if isinstance(task, Task):
                        self.task_pool.append(task)
                        logger.info(
                            "[INFO][RecursiveWorkflow] Added task"
                            f" {task} to workflow"
                        )
            elif isinstance(task, Task):
                self.task_pool.append(task)
                logger.info(
                    f"[INFO][RecursiveWorkflow] Added task {task} to"
                    " workflow"
                )
        except Exception as error:
            logger.warning(f"[ERROR][RecursiveWorkflow] {error}")
            raise error

    def run(self):
        """
        Executes the tasks in the workflow until the stop token is encountered.

        Returns:
            None
        """
        try:
            loop = 0
            while loop < self.max_loops:
                for task in self.task_pool:
                    while True:
                        result = task.run()
                        if (
                            result is not None
                            and self.stop_token in result
                        ):
                            break
                        print(f"{result}")
                loop += 1

            return result
        except Exception as error:
            logger.warning(f"[ERROR][RecursiveWorkflow] {error}")
            raise error
