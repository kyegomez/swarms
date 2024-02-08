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

In summary, the `RecursiveWorkflow` class is designed to automate tasks by adding and executing these tasks recursively until a stopping condition is reached. This can be achieved by utilizing the `add` and `run` methods provided. A general format for adding and utilizing the `RecursiveWorkflow` class has been provided under the "Examples" section. If you require any further information, view other sections, like Args and Source Code for specifics on using the class effectively.
