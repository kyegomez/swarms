import multiprocessing as mp
from typing import List, Optional

from swarms.structs.base import BaseStructure


class LoadBalancer(BaseStructure):
    """
    A load balancer class that distributes tasks among multiple workers.

    Args:
        num_workers (int): The number of worker processes to create.
        agents (Optional[List]): A list of agents to assign to the load balancer.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        num_workers (int): The number of worker processes.
        agents (Optional[List]): A list of agents assigned to the load balancer.
        tasks (List): A list of tasks to be executed.
        results (List): A list of results from the executed tasks.
        workers (List): A list of worker processes.

    Methods:
        add_task: Add a task to the load balancer.

    """

    def __init__(
        self,
        num_workers: int = 1,
        agents: Optional[List] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_workers = num_workers
        self.agents = agents
        self.tasks = []
        self.results = []
        self.workers = []
        self._init_workers()

    def _init_workers(self):
        for i in range(self.num_workers):
            worker = mp.Process(target=self._worker)
            worker.start()
            self.workers.append(worker)

    def _worker(self):
        while True:
            task = self._get_task()
            if task is None:
                break
            result = self._run_task(task)
            self._add_result(result)

    def _get_task(self):
        if len(self.tasks) == 0:
            return None
        return self.tasks.pop(0)

    def _run_task(self, task):
        return task()

    def _add_result(self, result):
        self.results.append(result)

    def add_task(self, task):
        self.tasks.append(task)
