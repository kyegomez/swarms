from typing import List, Callable, Union, Optional
from loguru import logger
from swarms.structs.base_swarm import BaseSwarm
from queue import PriorityQueue
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
import time
from pydantic import BaseModel, Field


class SwarmRunData(BaseModel):
    """
    Pydantic model to capture metadata about each swarm's execution.
    """

    swarm_name: str
    task: str
    priority: int
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status: str = "Pending"
    retries: int = 0
    result: Optional[str] = None
    exception: Optional[str] = None


class FederatedSwarmModel(BaseModel):
    """
    Pydantic base model to capture and log data for the FederatedSwarm system.
    """

    task: str
    swarms_data: List[SwarmRunData] = Field(default_factory=list)

    def add_swarm(self, swarm_name: str, task: str, priority: int):
        swarm_data = SwarmRunData(
            swarm_name=swarm_name, task=task, priority=priority
        )
        self.swarms_data.append(swarm_data)

    def update_swarm_status(
        self,
        swarm_name: str,
        status: str,
        start_time: float = None,
        end_time: float = None,
        retries: int = 0,
        result: str = None,
        exception: str = None,
    ):
        for swarm in self.swarms_data:
            if swarm.name == swarm_name:
                swarm.status = status
                if start_time:
                    swarm.start_time = start_time
                if end_time:
                    swarm.end_time = end_time
                    swarm.duration = end_time - swarm.start_time
                swarm.retries = retries
                swarm.result = result
                swarm.exception = exception
                break


class FederatedSwarm:
    def __init__(
        self,
        swarms: List[Union[BaseSwarm, Callable]],
        max_workers: int = 4,
    ):
        """
        Initializes the FederatedSwarm with a list of swarms or callable objects and
        sets up a priority queue and thread pool for concurrency.

        Args:
            swarms (List[Union[BaseSwarm, Callable]]): A list of swarms (BaseSwarm) or callable objects.
            max_workers (int): The maximum number of concurrent workers (threads) to run swarms in parallel.
        """
        self.swarms = PriorityQueue()
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.max_workers
        )
        self.task_queue = []
        self.future_to_swarm = {}
        self.results = {}
        self.validate_swarms(swarms)

    def init_metadata(self, task: str):
        """
        Initializes the Pydantic base model to capture metadata about the current task and swarms.
        """
        self.metadata = FederatedSwarmModel(task=task)
        for priority, swarm in list(self.swarms.queue):
            swarm_name = (
                swarm.__class__.__name__
                if hasattr(swarm, "__class__")
                else str(swarm)
            )
            self.metadata.add_swarm(
                swarm_name=swarm_name, task=task, priority=priority
            )
        logger.info(f"Metadata initialized for task '{task}'.")

    def validate_swarms(
        self, swarms: List[Union[BaseSwarm, Callable]]
    ):
        """
        Validates and adds swarms to the priority queue, ensuring each swarm has a `run(task)` method.

        Args:
            swarms (List[Union[BaseSwarm, Callable]]): List of swarms with an optional priority value.
        """
        for swarm, priority in swarms:
            if not callable(swarm):
                raise TypeError(f"{swarm} is not callable.")

            if hasattr(swarm, "run"):
                logger.info(f"{swarm} has a 'run' method.")
            else:
                raise AttributeError(
                    f"{swarm} does not have a 'run(task)' method."
                )

            self.swarms.put((priority, swarm))
            logger.info(
                f"Swarm {swarm} added with priority {priority}."
            )

    def run_parallel(
        self,
        task: str,
        timeout: Optional[float] = None,
        retries: int = 0,
    ):
        """
        Runs all swarms in parallel with prioritization and optional timeout.

        Args:
            task (str): The task to be passed to the `run` method of each swarm.
            timeout (Optional[float]): Maximum time allowed for each swarm to run.
            retries (int): Number of retries allowed for failed swarms.
        """
        logger.info(
            f"Running task '{task}' in parallel with timeout: {timeout}, retries: {retries}"
        )
        self.init_metadata(task)

        while not self.swarms.empty():
            priority, swarm = self.swarms.get()
            swarm_name = (
                swarm.__class__.__name__
                if hasattr(swarm, "__class__")
                else str(swarm)
            )
            future = self.thread_pool.submit(
                self._run_with_retry,
                swarm,
                task,
                retries,
                timeout,
                swarm_name,
            )
            self.future_to_swarm[future] = swarm

        for future in as_completed(self.future_to_swarm):
            swarm = self.future_to_swarm[future]
            try:
                result = future.result()
                swarm_name = (
                    swarm.__class__.__name__
                    if hasattr(swarm, "__class__")
                    else str(swarm)
                )
                self.metadata.update_swarm_status(
                    swarm_name=swarm_name,
                    status="Completed",
                    result=result,
                )
                logger.info(
                    f"Swarm {swarm_name} completed successfully."
                )
            except Exception as e:
                swarm_name = (
                    swarm.__class__.__name__
                    if hasattr(swarm, "__class__")
                    else str(swarm)
                )
                self.metadata.update_swarm_status(
                    swarm_name=swarm_name,
                    status="Failed",
                    exception=str(e),
                )
                logger.error(f"Swarm {swarm_name} failed: {e}")
                self.results[swarm] = "Failed"

    def run_sequentially(
        self,
        task: str,
        retries: int = 0,
        timeout: Optional[float] = None,
    ):
        """
        Runs all swarms sequentially in order of priority.

        Args:
            task (str): The task to pass to the `run` method of each swarm.
            retries (int): Number of retries for failed swarms.
            timeout (Optional[float]): Optional time limit for each swarm.
        """
        logger.info(f"Running task '{task}' sequentially.")

        while not self.swarms.empty():
            priority, swarm = self.swarms.get()
            try:
                logger.info(
                    f"Running swarm {swarm} with priority {priority}."
                )
                self._run_with_retry(swarm, task, retries, timeout)
                logger.info(f"Swarm {swarm} completed successfully.")
            except Exception as e:
                logger.error(f"Swarm {swarm} failed with error: {e}")

    def _run_with_retry(
        self,
        swarm: Union[BaseSwarm, Callable],
        task: str,
        retries: int,
        timeout: Optional[float],
        swarm_name: str,
    ):
        """
        Helper function to run a swarm with a retry mechanism and optional timeout.

        Args:
            swarm (Union[BaseSwarm, Callable]): The swarm to run.
            task (str): The task to pass to the swarm.
            retries (int): The number of retries allowed for the swarm in case of failure.
            timeout (Optional[float]): Maximum time allowed for the swarm to run.
            swarm_name (str): Name of the swarm (used for metadata).
        """
        attempts = 0
        start_time = time.time()
        while attempts <= retries:
            try:
                logger.info(
                    f"Running swarm {swarm}. Attempt: {attempts + 1}"
                )
                self.metadata.update_swarm_status(
                    swarm_name=swarm_name,
                    status="Running",
                    start_time=start_time,
                )
                if hasattr(swarm, "run"):
                    if timeout:
                        start_time = time.time()
                        swarm.run(task)
                        duration = time.time() - start_time
                        if duration > timeout:
                            raise TimeoutError(
                                f"Swarm {swarm} timed out after {duration:.2f}s."
                            )
                    else:
                        swarm.run(task)
                else:
                    swarm(task)
                end_time = time.time()
                self.metadata.update_swarm_status(
                    swarm_name=swarm_name,
                    status="Completed",
                    end_time=end_time,
                    retries=attempts,
                )
                return "Success"
            except Exception as e:
                logger.error(f"Swarm {swarm} failed: {e}")
                attempts += 1
                if attempts > retries:
                    end_time = time.time()
                    self.metadata.update_swarm_status(
                        swarm_name=swarm_name,
                        status="Failed",
                        end_time=end_time,
                        retries=attempts,
                        exception=str(e),
                    )
                    logger.error(f"Swarm {swarm} exhausted retries.")
                    raise

    def add_swarm(
        self, swarm: Union[BaseSwarm, Callable], priority: int
    ):
        """
        Adds a new swarm to the FederatedSwarm at runtime.

        Args:
            swarm (Union[BaseSwarm, Callable]): The swarm to add.
            priority (int): The priority level for the swarm.
        """
        self.swarms.put((priority, swarm))
        logger.info(
            f"Swarm {swarm} added dynamically with priority {priority}."
        )

    def queue_task(self, task: str):
        """
        Adds a task to the internal task queue for batch processing.

        Args:
            task (str): The task to queue.
        """
        self.task_queue.append(task)
        logger.info(f"Task '{task}' added to the queue.")

    def process_task_queue(self):
        """
        Processes all tasks in the task queue.
        """
        for task in self.task_queue:
            logger.info(f"Processing task: {task}")
            self.run_parallel(task)
        self.task_queue = []

    def log_swarm_results(self):
        """
        Logs the results of all swarms after execution.
        """
        logger.info("Logging swarm results...")
        for swarm, result in self.results.items():
            logger.info(f"Swarm {swarm}: {result}")

    def get_swarm_status(self) -> dict:
        """
        Retrieves the status of each swarm (completed, running, failed).

        Returns:
            dict: Dictionary containing swarm statuses.
        """
        status = {}
        for future, swarm in self.future_to_swarm.items():
            if future.done():
                status[swarm] = "Completed"
            elif future.running():
                status[swarm] = "Running"
            else:
                status[swarm] = "Failed"
        return status

    def cancel_running_swarms(self):
        """
        Cancels all currently running swarms by shutting down the thread pool.
        """
        logger.warning("Cancelling all running swarms...")
        self.thread_pool.shutdown(wait=False)
        logger.info("All running swarms cancelled.")


# Example Usage:


# class ExampleSwarm(BaseSwarm):
#     def run(self, task: str):
#         logger.info(f"ExampleSwarm is processing task: {task}")


# def example_callable(task: str):
#     logger.info(f"Callable is processing task: {task}")


# if __name__ == "__main__":
#     swarms = [(ExampleSwarm(), 1), (example_callable, 2)]
#     federated_swarm = FederatedSwarm(swarms)

#     # Run in parallel
#     federated_swarm.run_parallel(
#         "Process data", timeout=10, retries=3
#     )

#     # Run sequentially
#     federated_swarm.run_sequentially("Process data sequentially")

#     # Log results
#     federated_swarm.log_swarm_results()

#     # Get status of swarms
#     status = federated_swarm.get_swarm_status()
#     logger.info(f"Swarm statuses: {status}")

#     # Cancel running swarms (if needed)
#     # federated_swarm.cancel_running_swarms()
