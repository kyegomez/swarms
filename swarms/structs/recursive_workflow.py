from typing import List

from swarms.structs.base import BaseStruct
from swarms.structs.task import Task


class RecursiveWorkflow(BaseStruct):
    """
    RecursiveWorkflow class for running a task recursively until a stopping condition is met.

    Args:
        task (Task): The task to execute.
        stop_token (Any): The token that indicates when to stop the workflow.

    Attributes:
        task (Task): The task to execute.
        stop_token (Any): The token that indicates when to stop the workflow.

    Examples:
    >>> from swarms.models import OpenAIChat
    >>> from swarms.structs import RecursiveWorkflow, Task
    >>> llm = OpenAIChat(openai_api_key="")
    >>> task = Task(llm, "What's the weather in miami")
    >>> workflow = RecursiveWorkflow()
    >>> workflow.add(task)
    >>> workflow.run()
    """

    def __init__(self, stop_token: str = "<DONE>"):
        self.stop_token = stop_token
        self.tasks = List[Task]

        assert (
            self.stop_token is not None
        ), "stop_token cannot be None"

    def add(self, task: Task):
        assert task is not None, "task cannot be None"
        return self.tasks.appennd(task)

    def run(self):
        """
        Executes the tasks in the workflow until the stop token is encountered.

        Returns:
            None
        """
        try:
            for task in self.tasks:
                while True:
                    result = task.execute()
                    if self.stop_token in result:
                        break
        except Exception as error:
            print(f"[ERROR][RecursiveWorkflow] {error}")
            raise error
