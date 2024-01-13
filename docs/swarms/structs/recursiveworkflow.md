**Module/Function Name: RecursiveWorkflow**

`class` RecursiveWorkflow(BaseStructure):

Creates a recursive workflow structure for executing a task until a stated stopping condition is reached. 

#### Parameters
* *task* (`Task`): The task to execute.
* *stop_token* (`Any`): The token that signals the termination of the workflow.

#### Examples:
```python
from swarms.models import OpenAIChat
from swarms.structs import RecursiveWorkflow, Task

llm = OpenAIChat(openai_api_key="YourKey")
task = Task(llm, "What's the weather in miami")
workflow = RecursiveWorkflow(stop_token="<DONE>")
workflow.add(task)
workflow.run()
```

Returns: None

#### Source Code:

```python
class RecursiveWorkflow(BaseStructure):
    def __init__(self, stop_token: str = "<DONE>"):
        """
        Args:
            stop_token (str, optional): The token that indicates when to stop the workflow. Default is "<DONE>". 
            The stop_token indicates the value at which the current workflow is finished. 
        """
        self.stop_token = stop_token
        self.tasks = []

        assert (
            self.stop_token is not None
        ), "stop_token cannot be None"

    def add(self, task: Task, tasks: List[Task] = None):
        """Adds a task to the workflow. 
        Args:
            task (Task): The task to be added.
            tasks (List[Task], optional): List of tasks to be executed. 
        """
        try:
            if tasks:
                for task in tasks:
                    self.tasks.append(task)
            else:
                self.tasks.append(task)
        except Exception as error:
            print(f"[ERROR][ConcurrentWorkflow] {error}")
            raise error

    def run(self):
        """Executes the tasks in the workflow until the stop token is encountered"""
        try:
            for task in self.tasks:
                while True:
                    result = task.execute()
                    if self.stop_token in result:
                        break
        except Exception as error:
            print(f"[ERROR][RecursiveWorkflow] {error}")
            raise error
```

In summary, the `RecursiveWorkflow` class is designed to automate tasks by adding and executing these tasks recursively until a stopping condition is reached. This can be achieved by utilizing the `add` and `run` methods provided. A general format for adding and utilizing the `RecursiveWorkflow` class has been provided under the "Examples" section. If you require any further information, view other sections, like Args and Source Code for specifics on using the class effectively.
