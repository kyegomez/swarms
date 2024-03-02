import logging
from functools import wraps
from multiprocessing import Manager, Pool, cpu_count
from time import sleep
from typing import List

from swarms.structs.base_workflow import BaseWorkflow
from swarms.structs.task import Task

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Retry on failure
def retry_on_failure(max_retries: int = 3, delay: int = 5):
    """
    Decorator that retries a function a specified number of times on failure.

    Args:
        max_retries (int): The maximum number of retries (default: 3).
        delay (int): The delay in seconds between retries (default: 5).

    Returns:
        The result of the function if it succeeds within the maximum number of retries,
        otherwise None.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as error:
                    logging.error(
                        f"Error: {str(error)}, retrying in"
                        f" {delay} seconds..."
                    )
                    sleep(delay)
            return None

        return wrapper

    return decorator


class MultiProcessingWorkflow(BaseWorkflow):
    """
    Initialize a MultiProcessWorkflow object.

    Args:
        max_workers (int): The maximum number of workers to use for parallel processing.
        autosave (bool): Flag indicating whether to automatically save the workflow.
        tasks (List[Task]): A list of Task objects representing the workflow tasks.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Example:
    >>> from swarms.structs.multi_process_workflow import MultiProcessingWorkflow
    >>> from swarms.structs.task import Task
    >>> from datetime import datetime
    >>> from time import sleep
    >>>
    >>> # Define a simple task
    >>> def simple_task():
    >>>     sleep(1)
    >>>     return datetime.now()
    >>>
    >>> # Create a task object
    >>> task = Task(
    >>>     name="Simple Task",
    >>>     execute=simple_task,
    >>>     priority=1,
    >>> )
    >>>
    >>> # Create a workflow with the task
    >>> workflow = MultiProcessingWorkflow(tasks=[task])
    >>>
    >>> # Run the workflow
    >>> results = workflow.run(task)
    >>>
    >>> # Print the results
    >>> print(results)
    """

    def __init__(
        self,
        max_workers: int = 5,
        autosave: bool = True,
        tasks: List[Task] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_workers = max_workers
        self.autosave = autosave
        self.tasks = sorted(
            tasks or [], key=lambda task: task.priority, reverse=True
        )

        self.max_workers or cpu_count()

        if tasks is None:
            tasks = []

        self.tasks = tasks

    def execute_task(self, task: Task, *args, **kwargs):
        """Execute a task and handle exceptions.

        Args:
            task (Task): The task to execute.
            *args: Additional positional arguments for the task execution.
            **kwargs: Additional keyword arguments for the task execution.

        Returns:
            Any: The result of the task execution.

        """
        try:
            result = task.execute(*args, **kwargs)

            logging.info(
                f"Task {task} completed successfully with result"
                f" {result}"
            )

            if self.autosave:
                self._autosave_task_result(task, result)

        except Exception as e:
            logging.error(
                "An error occurred during execution of task"
                f" {task}: {str(e)}",
                exc_info=True,
            )
            return None

    def run(self, task: Task, *args, **kwargs):
        """Run the workflow.

        Args:
            task (Task): The task to run.
            *args: Additional positional arguments for the task execution.
            **kwargs: Additional keyword arguments for the task execution.

        Returns:
            List[Any]: The results of all executed tasks.

        """
        try:
            results = []
            with Manager() as manager:
                with Pool(
                    processes=self.max_workers, *args, **kwargs
                ) as pool:
                    # Using manager.list() to collect results in a process safe way
                    results_list = manager.list()
                    jobs = [
                        pool.apply_async(
                            self.execute_task,
                            (task,),
                            callback=results_list.append,
                            timeout=task.timeout,
                            *args,
                            **kwargs,
                        )
                        for task in self.tasks
                    ]

                    # Wait for all jobs to complete
                    for job in jobs:
                        job.get()

                    results = list(results_list)

                return results
        except Exception as error:
            logging.error(f"Error in run: {error}")
            return None

    def _autosave_task_result(self, task: Task, result):
        """Autosave task result. This should be adapted based on how autosaving is implemented.

        Args:
            task (Task): The task for which to autosave the result.
            result (Any): The result of the task execution.

        """
        # Note: This method might need to be adapted to ensure it's process-safe, depending on how autosaving is implemented.
        logging.info(f"Autosaving result for task {task}: {result}")
        # Actual autosave logic here
