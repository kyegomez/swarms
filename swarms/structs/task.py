from swarms.structs.agent import Agent

from typing import List, Dict, Any, Sequence


class Task:
    """
    Task is a unit of work that can be executed by a set of agents.

    A task is defined by a task name and a set of agents that can execute the task.
    The task can also have a set of dependencies, which are the names of other tasks
    that must be executed before this task can be executed.

    Args:
        id (str): A unique identifier for the task
        task (str): The name of the task
        agents (Sequence[Agent]): A list of agents that can execute the task
        dependencies (List[str], optional): A list of task names that must be executed before this task can be executed. Defaults to [].

    Methods:
        execute(parent_results: Dict[str, Any]): Executes the task by passing the results of the parent tasks to the agents.
    """

    def __init__(
        self,
        id: str,
        task: str,
        agents: Sequence[Agent],
        dependencies: List[str] = [],
    ):
        self.id = id
        self.task = task
        self.agents = agents
        self.dependencies = dependencies
        self.results = []

    def execute(self, parent_results: Dict[str, Any]):
        """Executes the task by passing the results of the parent tasks to the agents.

        Args:
            parent_results (Dict[str, Any]): _description_
        """
        args = [parent_results[dep] for dep in self.dependencies]
        for agent in self.agents:
            result = agent.run(self.task, *args)
            self.results.append(result)
            args = [
                result
            ]  # The output of one agent becomes the input to the next
