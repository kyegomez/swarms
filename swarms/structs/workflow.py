from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional


class Workflow:
    """
    Workflows are ideal for prescriptive processes that need to be executed
    sequentially.
    They string together multiple tasks of varying types, and can use Short-Term Memory
    or pass specific arguments downstream.


    Usage
    llm = LLM()
    workflow = Workflow(llm)

    workflow.add("What's the weather in miami")
    workflow.add("Provide detauls for {{ parent_output }}")
    workflow.add("Summarize the above information: {{ parent_output}})

    workflow.run()


    """

    class Task:
        def __init__(self, task: str):
            self.task = task
            self.parents = []
            self.children = []
            self.output = None
            self.structure = None

        def add_child(self, child: "Workflow.Task"):
            self.children.append(child)
            child.parents.append(self)
            child.structure = self.structure

        def execute(self) -> Any:
            prompt = self.task.replace(
                "{{ parent_input }}", self.parents[0].output if self.parents else ""
            )
            response = self.structure.agent.run(prompt)
            self.output = response
            return response

    def __init__(self, agent, parallel: bool = False):
        """__init__"""
        self.agent = agent
        self.tasks: List[Workflow.Task] = []
        self.parallel = parallel

    def add(self, task: str) -> Task:
        """Add a task"""
        task = self.Task(task)

        if self.last_task():
            self.last_task().add_child(task)
        else:
            task.structure = self
            self.tasks.append(task)
        return task

    def first_task(self) -> Optional[Task]:
        """Add first task"""
        return self.tasks[0] if self.tasks else None

    def last_task(self) -> Optional[Task]:
        """Last task"""
        return self.tasks[-1] if self.tasks else None

    def run(self, *args) -> Task:
        """Run tasks"""
        [task.reset() for task in self.tasks]

        if self.parallel:
            with ThreadPoolExecutor() as executor:
                list(executor.map(self.__run_from_task, [self.first_task]))
        else:
            self.__run_from_task(self.first_task())

        return self.last_task()

    def context(self, task: Task) -> Dict[str, Any]:
        """Context in tasks"""
        return {
            "parent_output": task.parents[0].output
            if task.parents and task.parents[0].output
            else None,
            "parent": task.parents[0] if task.parents else None,
            "child": task.children[0] if task.children else None,
        }

    def __run_from_task(self, task: Optional[Task]) -> None:
        """Run from task"""
        if task is None:
            return
        else:
            if isinstance(task.execute(), Exception):
                return
            else:
                self.__run_from_task(next(iter(task.children), None))
