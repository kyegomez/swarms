from swarms.models import OpenAIChat
from swarms.structs.flow import Flow

import concurrent.futures
from typing import Callable, List, Dict, Any, Sequence


class Task:
    def __init__(
        self,
        id: str,
        task: str,
        flows: Sequence[Flow],
        dependencies: List[str] = [],
    ):
        self.id = id
        self.task = task
        self.flows = flows
        self.dependencies = dependencies
        self.results = []

    def execute(self, parent_results: Dict[str, Any]):
        args = [parent_results[dep] for dep in self.dependencies]
        for flow in self.flows:
            result = flow.run(self.task, *args)
            self.results.append(result)
            args = [
                result
            ]  # The output of one flow becomes the input to the next


class Workflow:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.executor = concurrent.futures.ThreadPoolExecutor()

    def add_task(self, task: Task):
        self.tasks[task.id] = task

    def run(self):
        completed_tasks = set()
        while len(completed_tasks) < len(self.tasks):
            futures = []
            for task in self.tasks.values():
                if task.id not in completed_tasks and all(
                    dep in completed_tasks for dep in task.dependencies
                ):
                    future = self.executor.submit(
                        task.execute,
                        {
                            dep: self.tasks[dep].results
                            for dep in task.dependencies
                        },
                    )
                    futures.append((future, task.id))

            for future, task_id in futures:
                future.result()  # Wait for task completion
                completed_tasks.add(task_id)

    def get_results(self):
        return {task_id: task.results for task_id, task in self.tasks.items()}


# create flows
llm = OpenAIChat(openai_api_key="sk-")

flow1 = Flow(llm, max_loops=1)
flow2 = Flow(llm, max_loops=1)
flow3 = Flow(llm, max_loops=1)
flow4 = Flow(llm, max_loops=1)


# Create tasks with their respective Flows and task strings
task1 = Task("task1", "Generate a summary on Quantum field theory", [flow1])
task2 = Task(
    "task2",
    "Elaborate on the summary of topic X",
    [flow2, flow3],
    dependencies=["task1"],
)
task3 = Task(
    "task3", "Generate conclusions for topic X", [flow4], dependencies=["task1"]
)

# Create a workflow and add tasks
workflow = Workflow()
workflow.add_task(task1)
workflow.add_task(task2)
workflow.add_task(task3)

# Run the workflow
workflow.run()

# Get results
results = workflow.get_results()
print(results)
