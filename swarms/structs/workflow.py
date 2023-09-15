# from __future__ import annotations

# import concurrent.futures as futures
# import logging
# import uuid
# from abc import ABC, abstractmethod
# from graphlib import TopologicalSorter
# from logging import Logger
# from typing import Optional, Union

# from rich.logging import RichHandler

# # from swarms.artifacts.error_artifact import ErrorArtifact
# from swarms.artifacts.main import Artifact as ErrorArtifact
# from swarms.structs.task import BaseTask


# #@shapeless
# class Workflow(ABC):
#     def __init__(
#         self, 
#         id: str = uuid.uuid4().hex, 
#         model = None,
#         custom_logger: Optional[Logger] = None, 
#         logger_level: int = logging.INFO, 
#         futures_executor: futures.Executor = futures.ThreadPoolExecutor()
#     ):
#         self.id = id
#         self.model = model
#         self.custom_logger = custom_logger
#         self.logger_level = logger_level

#         self.futures_executor = futures_executor
#         self._execution_args = ()
#         self._logger = None

#         [task.preprocess(self) for task in self.tasks]

#         self.model.structure = self

#     @property
#     def execution_args(self) -> tuple:
#         return self._execution_args

#     @property
#     def logger(self) -> Logger:
#         if self.custom_logger:
#             return self.custom_logger
#         else:
#             if self._logger is None:
#                 self._logger = logging.getLogger(self.LOGGER_NAME)

#                 self._logger.propagate = False
#                 self._logger.level = self.logger_level

#                 self._logger.handlers = [
#                     RichHandler(
#                         show_time=True,
#                         show_path=False
#                     )
#                 ]
#             return self._logger

#     def is_finished(self) -> bool:
#         return all(s.is_finished() for s in self.tasks)

#     def is_executing(self) -> bool:
#         return any(s for s in self.tasks if s.is_executing())

#     def find_task(self, task_id: str) -> Optional[BaseTask]:
#         return next((task for task in self.tasks if task.id == task_id), None)

#     def add_tasks(self, *tasks: BaseTask) -> list[BaseTask]:
#         return [self.add_task(s) for s in tasks]

#     def context(self, task: BaseTask) -> dict[str, any]:
#         return {
#             "args": self.execution_args,
#             "structure": self,
#         }

#     @abstractmethod
#     def add(self, task: BaseTask) -> BaseTask:
#         task.preprocess(self)
#         self.tasks.append(task)
#         return task

#     @abstractmethod
#     def run(self, *args) -> Union[BaseTask, list[BaseTask]]:
#         self._execution_args = args
#         ordered_tasks = self.order_tasks()
#         exit_loop = False

#         while not self.is_finished() and not exit_loop:
#             futures_list = {}

#             for task in ordered_tasks:
#                 if task.can_execute():
#                     future = self.futures_executor.submit(task.execute)
#                     futures_list[future] = task

#             # Wait for all tasks to complete
#             for future in futures.as_completed(futures_list):
#                 if isinstance(future.result(), ErrorArtifact):
#                     exit_loop = True
#                     break

#         self._execution_args = ()

#         return self.output_tasks()

#     def context(self, task: BaseTask) -> dict[str, any]:
#         context = super().context(task)

#         context.update(
#             {
#                 "parent_outputs": {parent.id: parent.output.to_text() if parent.output else "" for parent in task.parents},
#                 "parents": {parent.id: parent for parent in task.parents},
#                 "children": {child.id: child for child in task.children}
#             }
#         )

#         return context

#     def output_tasks(self) -> list[BaseTask]:
#         return [task for task in self.tasks if not task.children]

#     def to_graph(self) -> dict[str, set[str]]:
#         graph: dict[str, set[str]] = {}

#         for key_task in self.tasks:
#             graph[key_task.id] = set()

#             for value_task in self.tasks:
#                 if key_task.id in value_task.child_ids:
#                     graph[key_task.id].add(value_task.id)

#         return graph

#     def order_tasks(self) -> list[BaseTask]:
#         return [self.find_task(task_id) for task_id in TopologicalSorter(self.to_graph()).static_order()]


from __future__ import annotations

from typing import Any, Dict, List, Optional

from swarms.artifacts.error_artifacts import ErrorArtifact
from swarms.structs.task import BaseTask

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
        llm
    ):
        self.llm = llm
        self.tasks: List[BaseTask] = []
    
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


