from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from swarms.artifacts.error_artifacts import ErrorArtifact
from swarms.structs.task import BaseTask
import concurrent.futures

class StringTask(BaseTask):
    def __init__(
        self,
        task
    ):
        super().__init__()
        self.task = task
    
    def execute(self) -> Any:
        prompt = self.task_string.replace("{{ parent_input }}", self.parents[0].output if self.parents else "")
        response = self.structure.llm.run(prompt)
        self.output = response
        return response



class Workflow:
    """
    Workflows are ideal for prescriptive processes that need to be executed sequentially. 
    They string together multiple tasks of varying types, and can use Short-Term Memory 
    or pass specific arguments downstream.



    ```
    llm = LLM()
    workflow = Workflow(llm)

    workflow.add("What's the weather in miami")
    workflow.add("Provide detauls for {{ parent_output }}")
    workflow.add("Summarize the above information: {{ parent_output}})

    workflow.run()


    """
    def __init__(
        self,
        llm,
        parallel: bool = False
    ):
        self.llm = llm
        self.tasks: List[BaseTask] = []
        self.parallel = parallel

    
    def add(
        self,
        task: BaseTask
    ) -> BaseTask:
        task = StringTask(task)

        if self.last_task():
            self.last_task().add_child(task)
        else:
            task.structure = self
            self.tasks.append(task)
        return task

    def first_task(self) -> Optional[BaseTask]:
        return self.tasks[0] if self.tasks else None
    
    def last_task(self) -> Optional[BaseTask]:
        return self.tasks[-1] if self.tasks else None

    def run(self, *args) -> BaseTask:
        self._execution_args = args

        [task.reset() for task in self.tasks]

        if self.parallel:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                list(executor.map(self.__run_from_task, [self.first_task]))
        else:
            self.__run_from_task(self.first_task())

        self._execution_args = ()

        return self.last_task()

    
    def context(self, task: BaseTask) -> Dict[str, Any]:
        context = super().context(task)

        context.update(
            {
                "parent_output": task.parents[0].output.to_text() \
                    if task.parents and task.parents[0].output else None,
                "parent": task.parents[0] if task.parents else None,
                "child": task.children[0] if task.children else None
            }
        )
        return context

    
    def __run_from_task(self, task: Optional[BaseTask]) -> None:
        if task is None:
            return
        else:
            if isinstance(task.execute(), ErrorArtifact):
                return
            else:
                self.__run_from_task(next(iter(task.children), None))

