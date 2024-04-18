from swarms.structs.base_structure import BaseStructure
from swarms.structs.task import Task
from swarms.utils.logger import logger  # noqa: F401


class NonlinearWorkflow(BaseStructure):
    """
    Represents a Directed Acyclic Graph (DAG) workflow.

    Attributes:
        tasks (dict): A dictionary mapping task names to Task objects.
        edges (dict): A dictionary mapping task names to a list of dependencies.

    Methods:
        add(task: Task, *dependencies: str): Adds a task to the workflow with its dependencies.
        run(): Executes the workflow by running tasks in topological order.

    Examples:
    >>> from swarms.models import OpenAIChat
    >>> from swarms.structs import NonlinearWorkflow, Task
    >>> llm = OpenAIChat(openai_api_key="")
    >>> task = Task(llm, "What's the weather in miami")
    >>> workflow = NonlinearWorkflow()
    >>> workflow.add(task)
    >>> workflow.run()

    """

    def __init__(self, stopping_token: str = "<DONE>"):
        self.tasks = {}
        self.edges = {}
        self.stopping_token = stopping_token

    def add(self, task: Task, *dependencies: str):
        """
        Adds a task to the workflow with its dependencies.

        Args:
            task (Task): The task to be added.
            dependencies (str): Variable number of dependency task names.

        Raises:
            AssertionError: If the task is None.

        Returns:
            None
        """
        assert task is not None, "Task cannot be None"
        self.tasks[task.name] = task
        self.edges[task.name] = list(dependencies)
        logger.info(f"[NonlinearWorkflow] [Added task {task.name}]")

    def run(self):
        """
        Executes the workflow by running tasks in topological order.

        Raises:
            Exception: If a circular dependency is detected.

        Returns:
            None
        """
        try:
            # Create a copy of the edges
            edges = self.edges.copy()

            while edges:
                # Get all tasks with no dependencies
                ready_tasks = [
                    task for task, deps in edges.items() if not deps
                ]

                if not ready_tasks:
                    raise Exception("Circular dependency detected")

                # Run all ready tasks
                for task in ready_tasks:
                    result = self.tasks[task].execute()
                    if result == self.stopping_token:
                        return
                    del edges[task]

                # Remove dependencies on the ready tasks
                for deps in edges.values():
                    for task in ready_tasks:
                        if task in deps:
                            deps.remove(task)
        except Exception as error:
            logger.error(f"[ERROR][NonlinearWorkflow] {error}")
            raise error
