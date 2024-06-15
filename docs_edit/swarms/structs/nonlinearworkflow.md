
#### Class Name: NonlinearWorkflow

This class represents a Directed Acyclic Graph (DAG) workflow used to store tasks and their dependencies in a workflow. The structures can validate, execute and store the order of tasks present in the workflow. It has the following attributes and methods:

#### Attributes:
- `tasks` (dict): A dictionary mapping task names to Task objects.
- `edges` (dict): A dictionary mapping task names to a list of dependencies.
- `stopping_token` (str): The token which denotes the end condition for the workflow execution. Default: `<DONE>`

#### Methods:

1. `__init__(self, stopping_token: str = "<DONE>")`: The initialization method that sets up the NonlinearWorkflow object with an optional stopping token. This token marks the end of the workflow. 
  - **Args**:
    - `stopping_token` (str): The token to denote the end condition for the workflow execution.
  
2. `add(task: Task, *dependencies: str)`: Adds a task to the workflow along with its dependencies. This method is used to add a new task to the workflow with an optional list of dependency tasks.
  - **Args**:
    - `task` (Task): The task to be added.
    - `dependencies` (varargs): Variable number of dependency task names.
  - **Returns**: None

3. `run()`: This method runs the workflow by executing tasks in topological order. It runs the tasks according to the sequence of dependencies.
  - **Raises**:
    - `Exception`: If a circular dependency is detected.
  - **Returns**: None

#### Examples:

Usage Example 1:

```python
from swarms.models import OpenAIChat
from swarms.structs import NonlinearWorkflow, Task

# Initialize the OpenAIChat model
llm = OpenAIChat(openai_api_key="")
# Create a new Task
task = Task(llm, "What's the weather in Miami")
# Initialize the NonlinearWorkflow
workflow = NonlinearWorkflow()
# Add task to the workflow
workflow.add(task)
# Execute the workflow
workflow.run()
```

Usage Example 2:

```python
from swarms.models import OpenAIChat
from swarms.structs import NonlinearWorkflow, Task

# Initialize the OpenAIChat model
llm = OpenAIChat(openai_api_key="")
# Create new Tasks
task1 = Task(llm, "What's the weather in Miami")
task2 = Task(llm, "Book a flight to New York")
task3 = Task(llm, "Find a hotel in Paris")
# Initialize the NonlinearWorkflow
workflow = NonlinearWorkflow()
# Add tasks to the workflow with dependencies
workflow.add(task1, task2.name)
workflow.add(task2, task3.name)
workflow.add(task3, "OpenAIChat Initialization")
# Execute the workflow
workflow.run()
```

Usage Example 3:

```python
from swarms.models import OpenAIChat
from swarms.structs import NonlinearWorkflow, Task

# Initialize the OpenAIChat model
llm = OpenAIChat(openai_api_key="")
# Create new Tasks
task1 = Task(llm, "What's the weather in Miami")
task2 = Task(llm, "Book a flight to New York")
task3 = Task(llm, "Find a hotel in Paris")
# Initialize the NonlinearWorkflow
workflow = NonlinearWorkflow()
# Add tasks to the workflow with dependencies
workflow.add(task1)
workflow.add(task2, task1.name)
workflow.add(task3, task1.name, task2.name)
# Execute the workflow
workflow.run()
```

These examples illustrate the three main types of usage for the NonlinearWorkflow class and how it can be used to represent a directed acyclic graph (DAG) workflow with tasks and their dependencies.

---

The explanatory documentation details the architectural aspects, methods, attributes, examples, and usage patterns for the `NonlinearWorkflow` class. By following the module and function definition structure, the documentation provides clear and comprehensive descriptions of the class and its functionalities.
