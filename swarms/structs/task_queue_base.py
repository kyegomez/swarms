import threading
from abc import ABC

from swarms.structs.agent import Agent
from swarms.structs.task import Task


def synchronized_queue(method):
    """
    Decorator that synchronizes access to the decorated method using a lock.
    The lock is acquired before executing the method and released afterwards.

    Args:
        method: The method to be decorated.

    Returns:
        The decorated method.
    """
    timeout_sec = 5

    def wrapper(self, *args, **kwargs):
        with self.lock:
            self.lock.acquire(timeout=timeout_sec)
            try:
                return method(self, *args, **kwargs)
            except Exception as e:
                print(f"Failed to execute {method.__name__}: {e}")
            finally:
                self.lock.release()

    return wrapper


class TaskQueueBase(ABC):
    def __init__(self):
        self.lock = threading.Lock()

    @synchronized_queue
    # @abstractmethod
    def add(self, task: Task) -> bool:
        """Adds a task to the queue.

        Args:
            task (Task): The task to be added to the queue.

        Returns:
            bool: True if the task was successfully added, False otherwise.
        """
        ...

    @synchronized_queue
    # @abstractmethod
    def get(self, agent: Agent) -> Task:
        """Gets the next task from the queue.

        Args:
            agent (Agent): The agent requesting the task.

        Returns:
            Task: The next task from the queue.
        """
        ...

    @synchronized_queue
    # @abstractmethod
    def complete_task(self, task_id: str):
        """Sets the task as completed.

        Args:
            task_id (str): The ID of the task to be marked as completed.
        """
        ...

    @synchronized_queue
    # @abstractmethod
    def reset(self, task_id: str):
        """Resets the task if the agent failed to complete it.

        Args:
            task_id (str): The ID of the task to be reset.
        """
        ...
