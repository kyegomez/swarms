from concurrent.futures import ThreadPoolExecutor, as_completed
from graphlib import TopologicalSorter
from typing import Dict, List


class Task:
    """
    Task is a unit of work that can be executed by an agent
    """

    def __init__(self,
                 id: str,
                 parents: List["Task"] = None,
                 children: List["Task"] = None):
        self.id = id
        self.parents = parents
        self.children = children

    def can_execute(self):
        """
        can_execute returns True if the task can be executed
        """
        raise NotImplementedError

    def execute(self):
        """
        Execute the task

        """
        raise NotImplementedError


class NonLinearWorkflow:
    """
    NonLinearWorkflow constructs a non sequential DAG of tasks to be executed by agents


    Architecture:
    NonLinearWorkflow = Task + Agent + Executor

    ASCII Diagram:
    +-------------------+
    | NonLinearWorkflow |
    +-------------------+
    |                   |
    |                   |
    |                   |
    |                   |
    |                   |
    |                   |
    |                   |
    |                   |
    |                   |
    |                   |
    +-------------------+


    """

    def __init__(self, agents, iters_per_task):
        """A workflow is a collection of tasks that can be executed in parallel or sequentially."""
        super().__init__()
        self.executor = ThreadPoolExecutor()
        self.agents = agents
        self.tasks = []

    def add(self, task: Task):
        """Add a task to the workflow"""
        assert isinstance(task, Task), "Input must be an nstance of Task"
        self.tasks.append(task)
        return task

    def run(self):
        """Run the workflow"""
        ordered_tasks = self.ordered_tasks
        exit_loop = False

        while not self.is_finished() and not exit_loop:
            futures_list = {}

            for task in ordered_tasks:
                if task.can_execute:
                    future = self.executor.submit(self.agents.run,
                                                  task.task_string)
                    futures_list[future] = task

            for future in as_completed(futures_list):
                if isinstance(future.result(), Exception):
                    exit_loop = True
                    break
        return self.output_tasks()

    def output_tasks(self) -> List[Task]:
        """Output tasks from the workflow"""
        return [task for task in self.tasks if not task.children]

    def to_graph(self) -> Dict[str, set[str]]:
        """Convert the workflow to a graph"""
        graph = {
            task.id: set(child.id for child in task.children)
            for task in self.tasks
        }
        return graph

    def order_tasks(self) -> List[Task]:
        """Order the tasks USING TOPOLOGICAL SORTING"""
        task_order = TopologicalSorter(self.to_graph()).static_order()
        return [self.find_task(task_id) for task_id in task_order]
