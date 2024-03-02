import logging
import queue
import threading
from concurrent.futures import (
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    wait,
)
from typing import List

from swarms.structs.base_workflow import BaseWorkflow
from swarms.structs.task import Task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class PriorityTask:
    """
    Represents a task with a priority level.

    Attributes:
        task (Task): The task to be executed.
        priority (int): The priority level of the task.
    """

    def __init__(self, task: Task, priority: int = 0):
        self.task = task
        self.priority = priority

    def __lt__(self, other):
        return self.priority < other.priority


class MultiThreadedWorkflow(BaseWorkflow):
    """
    Represents a multi-threaded workflow that executes tasks concurrently using a thread pool.

    Args:
        max_workers (int): The maximum number of worker threads in the thread pool. Default is 5.
        autosave (bool): Flag indicating whether to automatically save task results. Default is True.
        tasks (List[PriorityTask]): List of priority tasks to be executed. Default is an empty list.
        retry_attempts (int): The maximum number of retry attempts for failed tasks. Default is 3.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        max_workers (int): The maximum number of worker threads in the thread pool.
        autosave (bool): Flag indicating whether to automatically save task results.
        retry_attempts (int): The maximum number of retry attempts for failed tasks.
        tasks_queue (PriorityQueue): The queue that holds the priority tasks.
        lock (Lock): The lock used for thread synchronization.

    Methods:
        execute_tasks: Executes the tasks in the thread pool and returns the results.
        _autosave_task_result: Autosaves the result of a task.

    """

    def __init__(
        self,
        max_workers: int = 5,
        autosave: bool = True,
        tasks: List[PriorityTask] = None,
        retry_attempts: int = 3,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_workers = max_workers
        self.autosave = autosave
        self.retry_attempts = retry_attempts
        if tasks is None:
            tasks = []
        self.tasks_queue = queue.PriorityQueue()
        for task in tasks:
            self.tasks_queue.put(task)
        self.lock = threading.Lock()

    def run(self):
        """
        Executes the tasks in the thread pool and returns the results.

        Returns:
            List: The list of results from the executed tasks.

        """
        results = []
        with ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_task = {}
            for _ in range(self.tasks_queue.qsize()):
                priority_task = self.tasks_queue.get_nowait()
                future = executor.submit(priority_task.task.execute)
                future_to_task[future] = (
                    priority_task.task,
                    0,
                )  # (Task, attempt)

            while future_to_task:
                # Wait for the next future to complete
                done, _ = wait(
                    future_to_task.keys(), return_when=FIRST_COMPLETED
                )

                for future in done:
                    task, attempt = future_to_task.pop(future)
                    try:
                        result = future.result()
                        results.append(result)
                        logging.info(
                            f"Task {task} completed successfully with"
                            f" result: {result}"
                        )
                        if self.autosave:
                            self._autosave_task_result(task, result)
                    except Exception as e:
                        logging.error(
                            f"Attempt {attempt+1} failed for task"
                            f" {task}: {str(e)}",
                            exc_info=True,
                        )
                        if attempt + 1 < self.retry_attempts:
                            # Retry the task
                            retry_future = executor.submit(
                                task.execute
                            )
                            future_to_task[retry_future] = (
                                task,
                                attempt + 1,
                            )
                        else:
                            logging.error(
                                f"Task {task} failed after"
                                f" {self.retry_attempts} attempts."
                            )

        return results

    def _autosave_task_result(self, task: Task, result):
        """
        Autosaves the result of a task.

        Args:
            task (Task): The task whose result needs to be autosaved.
            result: The result of the task.

        """
        with self.lock:
            logging.info(
                f"Autosaving result for task {task}: {result}"
            )
            # Actual autosave logic goes here
