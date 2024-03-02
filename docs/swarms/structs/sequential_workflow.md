# `SequentialWorkflow` Documentation

The **SequentialWorkflow** class is a Python module designed to facilitate the execution of a sequence of tasks in a sequential manner. It is a part of the `swarms.structs` package and is particularly useful for orchestrating the execution of various callable objects, such as functions or models, in a predefined order. This documentation will provide an in-depth understanding of the **SequentialWorkflow** class, including its purpose, architecture, usage, and examples.

## Purpose and Relevance

The **SequentialWorkflow** class is essential for managing and executing a series of tasks or processes, where each task may depend on the outcome of the previous one. It is commonly used in various application scenarios, including but not limited to:

1. **Natural Language Processing (NLP) Workflows:** In NLP workflows, multiple language models are employed sequentially to process and generate text. Each model may depend on the results of the previous one, making sequential execution crucial.

2. **Data Analysis Pipelines:** Data analysis often involves a series of tasks such as data preprocessing, transformation, and modeling steps. These tasks must be performed sequentially to ensure data consistency and accuracy.

3. **Task Automation:** In task automation scenarios, there is a need to execute a series of automated tasks in a specific order. Sequential execution ensures that each task is performed in a predefined sequence, maintaining the workflow's integrity.

By providing a structured approach to managing these tasks, the **SequentialWorkflow** class helps developers streamline their workflow execution and improve code maintainability.

## Key Concepts and Terminology

Before delving into the details of the **SequentialWorkflow** class, let's define some key concepts and terminology that will be used throughout the documentation:

### Task

A **task** refers to a specific unit of work that needs to be executed as part of the workflow. Each task is associated with a description and can be implemented as a callable object, such as a function or a model.

### Agent

A **agent** represents a callable object that can be a task within the **SequentialWorkflow**. Agents encapsulate the logic and functionality of a particular task. Agents can be functions, models, or any callable object that can be executed.

### Sequential Execution

Sequential execution refers to the process of running tasks one after the other in a predefined order. In a **SequentialWorkflow**, tasks are executed sequentially, meaning that each task starts only after the previous one has completed.

### Workflow

A **workflow** is a predefined sequence of tasks that need to be executed in a specific order. It represents the overall process or pipeline that the **SequentialWorkflow** manages.

### Dashboard (Optional)

A **dashboard** is an optional feature of the **SequentialWorkflow** that provides real-time monitoring and visualization of the workflow's progress. It displays information such as the current task being executed, task results, and other relevant metadata.

### Max Loops

The **maximum number of times** the entire workflow can be run. This parameter allows developers to control how many times the workflow is executed.

### Autosaving

**Autosaving** is a feature that allows the **SequentialWorkflow** to automatically save its state to a file at specified intervals. This feature helps in resuming a workflow from where it left off, even after interruptions.

Now that we have a clear understanding of the key concepts and terminology, let's explore the architecture and usage of the **SequentialWorkflow** class in more detail.

## Architecture of SequentialWorkflow

The architecture of the **SequentialWorkflow** class is designed to provide a structured and flexible way to define, manage, and execute a sequence of tasks. It comprises the following core components:

1. **Task**: The **Task** class represents an individual unit of work within the workflow. Each task has a description, which serves as a human-readable identifier for the task. Tasks can be implemented as callable objects, allowing for great flexibility in defining their functionality.

2. **Workflow**: The **SequentialWorkflow** class itself represents the workflow. It manages a list of tasks in the order they should be executed. Workflows can be run sequentially or asynchronously, depending on the use case.

3. **Task Execution**: Task execution is the process of running each task in the workflow. Tasks are executed one after another in the order they were added to the workflow. Task results can be passed as inputs to subsequent tasks.

4. **Dashboard (Optional)**: The **SequentialWorkflow** optionally includes a dashboard feature. The dashboard provides a visual interface for monitoring the progress of the workflow. It displays information about the current task, task results, and other relevant metadata.

5. **State Management**: The **SequentialWorkflow** supports state management, allowing developers to save and load the state of the workflow to and from JSON files. This feature is valuable for resuming workflows after interruptions or for sharing workflow configurations.

## Usage of SequentialWorkflow

The **SequentialWorkflow** class is versatile and can be employed in a wide range of applications. Its usage typically involves the following steps:

1. **Initialization**: Begin by initializing any callable objects or flows that will serve as tasks in the workflow. These callable objects can include functions, models, or any other Python objects that can be executed.

2. **Workflow Creation**: Create an instance of the **SequentialWorkflow** class. Specify the maximum number of loops the workflow should run and whether a dashboard should be displayed.

3. **Task Addition**: Add tasks to the workflow using the `add` method. Each task should be described using a human-readable description, and the associated agent (callable object) should be provided. Additional arguments and keyword arguments can be passed to the task.

4. **Task Execution**: Execute the workflow using the `run` method. The tasks within the workflow will be executed sequentially, with task results passed as inputs to subsequent tasks.

5. **Accessing Results**: After running the workflow, you can access the results of each task using the `get_task_results` method or by directly accessing the `result` attribute of each task.

6. **Optional Features**: Optionally, you can enable features such as autosaving of the workflow state and utilize the dashboard for real-time monitoring.


## Installation

Before using the Sequential Workflow library, you need to install it. You can install it via pip:

```bash
pip3 install --upgrade swarms
```

## Quick Start

Let's begin with a quick example to demonstrate how to create and run a Sequential Workflow. In this example, we'll create a workflow that generates a 10,000-word blog on "health and wellness" using an AI model and then summarizes the generated content.

```python
from swarms.models import OpenAIChat
from swarms.structs import Agent
from swarms.structs.sequential_workflow import SequentialWorkflow

# Initialize the language model agent (e.g., GPT-3)
llm = OpenAIChat(
    openai_api_key="YOUR_API_KEY",
    temperature=0.5,
    max_tokens=3000,
)

# Initialize flows for individual tasks
flow1 = Agent(llm=llm, max_loops=1, dashboard=False)
flow2 = Agent(llm=llm, max_loops=1, dashboard=False)

# Create the Sequential Workflow
workflow = SequentialWorkflow(max_loops=1)

# Add tasks to the workflow
workflow.add("Generate a 10,000 word blog on health and wellness.", flow1)
workflow.add("Summarize the generated blog", flow2)

# Run the workflow
workflow.run()

# Output the results
for task in workflow.tasks:
    print(f"Task: {task.description}, Result: {task.result}")
```

This quick example demonstrates the basic usage of the Sequential Workflow. It creates two tasks and executes them sequentially.

## Class: `Task`

### Description

The `Task` class represents an individual task in the workflow. A task is essentially a callable object, such as a function or a class, that can be executed sequentially. Tasks can have arguments and keyword arguments.

### Class Definition

```python
class Task:
    def __init__(self, description: str, agent: Union[Callable, Agent], args: List[Any] = [], kwargs: Dict[str, Any] = {}, result: Any = None, history: List[Any] = [])
```

### Parameters

- `description` (str): A description of the task.
- `agent` (Union[Callable, Agent]): The callable object representing the task. It can be a function, class, or a `Agent` instance.
- `args` (List[Any]): A list of positional arguments to pass to the task when executed. Default is an empty list.
- `kwargs` (Dict[str, Any]): A dictionary of keyword arguments to pass to the task when executed. Default is an empty dictionary.
- `result` (Any): The result of the task's execution. Default is `None`.
- `history` (List[Any]): A list to store the historical results of the task. Default is an empty list.

### Methods

#### `execute()`

Execute the task.

```python
def execute(self):
```

This method executes the task and updates the `result` and `history` attributes of the task. It checks if the task is a `Agent` instance and if the 'task' argument is needed.

## Class: `SequentialWorkflow`

### Description

The `SequentialWorkflow` class is responsible for managing a sequence of tasks and executing them in a sequential order. It provides methods for adding tasks, running the workflow, and managing the state of the tasks.

### Class Definition

```python
class SequentialWorkflow:
    def __init__(self, max_loops: int = 1, autosave: bool = False, saved_state_filepath: Optional[str] = "sequential_workflow_state.json", restore_state_filepath: Optional[str] = None, dashboard: bool = False, tasks: List[Task] = [])
```

### Parameters

- `max_loops` (int): The maximum number of times to run the workflow sequentially. Default is `1`.
- `autosave` (bool): Whether to enable autosaving of the workflow state. Default is `False`.
- `saved_state_filepath` (Optional[str]): The file path to save the workflow state when autosave is enabled. Default is `"sequential_workflow_state.json"`.
- `restore_state_filepath` (Optional[str]): The file path to restore the workflow state when initializing. Default is `None`.
- `dashboard` (bool): Whether to display a dashboard with workflow information. Default is `False`.
- `tasks` (List[Task]): A list of `Task` instances representing the tasks in the workflow. Default is an empty list.

### Methods

#### `add(task: str, agent: Union[Callable, Agent], *args, **kwargs)`

Add a task to the workflow.

```python
def add(self, task: str, agent: Union[Callable, Agent], *args, **kwargs) -> None:
```

This method adds a new task to the workflow. You can provide a description of the task, the callable object (function, class, or `Agent` instance), and any additional positional or keyword arguments required for the task.

#### `reset_workflow()`

Reset the workflow by clearing the results of each task.

```python
def reset_workflow(self) -> None:
```

This method clears the results of each task in the workflow, allowing you to start fresh without reinitializing the workflow.

#### `get_task_results()`

Get the results of each task in the workflow.

```python
def get_task_results(self) -> Dict[str, Any]:
```

This method returns a dictionary containing the results of each task in the workflow, where the keys are task descriptions, and the values are the corresponding results.

#### `remove_task(task_description: str)`

Remove a task from the workflow.

```python
def remove_task(self, task_description: str) -> None:
```

This method removes a specific task from the workflow based on its description.

#### `update_task(task_description: str, **updates)`

Update the arguments of a task in the workflow.

```python
def update_task(self, task_description: str, **updates) -> None:
```

This method allows you to update the arguments and keyword arguments of a task in the workflow. You specify the task's description and provide the updates as keyword arguments.

#### `save_workflow_state(filepath: Optional[str] = "sequential_workflow_state.json", **kwargs)`

Save the workflow state to a JSON file.

```python
def save_workflow_state(self, filepath: Optional[str] = "sequential_workflow_state.json", **kwargs) -> None:
```

This method saves the current state of the workflow, including the results and history of each task, to a JSON file. You can specify the file path for saving the state.

#### `load_workflow_state(filepath: str = None, **kwargs)`

Load the workflow state from a JSON file and restore the workflow state.

```python
def load_workflow_state(self, filepath: str = None, **kwargs) -> None:
```

This method loads a previously saved workflow state from a JSON file

 and restores the state, allowing you to continue the workflow from where it was saved. You can specify the file path for loading the state.

#### `run()`

Run the workflow sequentially.

```python
def run(self) -> None:
```

This method executes the tasks in the workflow sequentially. It checks if a task is a `Agent` instance and handles the agent of data between tasks accordingly.

#### `arun()`

Asynchronously run the workflow.

```python
async def arun(self) -> None:
```

This method asynchronously executes the tasks in the workflow sequentially. It's suitable for use cases where asynchronous execution is required. It also handles data agent between tasks.

#### `workflow_bootup(**kwargs)`

Display a bootup message for the workflow.

```python
def workflow_bootup(self, **kwargs) -> None:
```

This method displays a bootup message when the workflow is initialized. You can customize the message by providing additional keyword arguments.

#### `workflow_dashboard(**kwargs)`

Display a dashboard for the workflow.

```python
def workflow_dashboard(self, **kwargs) -> None:
```

This method displays a dashboard with information about the workflow, such as the number of tasks, maximum loops, and autosave settings. You can customize the dashboard by providing additional keyword arguments.

## Examples

Let's explore some examples to illustrate how to use the Sequential Workflow library effectively.

Sure, I'll recreate the usage examples section for each method and use case using the provided foundation. Here are the examples:

### Example 1: Adding Tasks to a Sequential Workflow

In this example, we'll create a Sequential Workflow and add tasks to it.

```python
from swarms.models import OpenAIChat
from swarms.structs import Agent
from swarms.structs.sequential_workflow import SequentialWorkflow

# Example usage
api_key = ""  # Your actual API key here

# Initialize the language agent
llm = OpenAIChat(
    openai_api_key=api_key,
    temperature=0.5,
    max_tokens=3000,
)

# Initialize Agents for individual tasks
flow1 = Agent(llm=llm, max_loops=1, dashboard=False)
flow2 = Agent(llm=llm, max_loops=1, dashboard=False)

# Create the Sequential Workflow
workflow = SequentialWorkflow(max_loops=1)

# Add tasks to the workflow
workflow.add("Generate a 10,000 word blog on health and wellness.", flow1)
workflow.add("Summarize the generated blog", flow2)

# Output the list of tasks in the workflow
print("Tasks in the workflow:")
for task in workflow.tasks:
    print(f"Task: {task.description}")
```

In this example, we create a Sequential Workflow and add two tasks to it.

### Example 2: Resetting a Sequential Workflow

In this example, we'll create a Sequential Workflow, add tasks to it, and then reset it.

```python
from swarms.models import OpenAIChat
from swarms.structs import Agent
from swarms.structs.sequential_workflow import SequentialWorkflow

# Example usage
api_key = ""  # Your actual API key here

# Initialize the language agent
llm = OpenAIChat(
    openai_api_key=api_key,
    temperature=0.5,
    max_tokens=3000,
)

# Initialize Agents for individual tasks
flow1 = Agent(llm=llm, max_loops=1, dashboard=False)
flow2 = Agent(llm=llm, max_loops=1, dashboard=False)

# Create the Sequential Workflow
workflow = SequentialWorkflow(max_loops=1)

# Add tasks to the workflow
workflow.add("Generate a 10,000 word blog on health and wellness.", flow1)
workflow.add("Summarize the generated blog", flow2)

# Reset the workflow
workflow.reset_workflow()

# Output the list of tasks in the workflow after resetting
print("Tasks in the workflow after resetting:")
for task in workflow.tasks:
    print(f"Task: {task.description}")
```

In this example, we create a Sequential Workflow, add two tasks to it, and then reset the workflow, clearing all task results.

### Example 3: Getting Task Results from a Sequential Workflow

In this example, we'll create a Sequential Workflow, add tasks to it, run the workflow, and then retrieve the results of each task.

```python
from swarms.models import OpenAIChat
from swarms.structs import Agent
from swarms.structs.sequential_workflow import SequentialWorkflow

# Example usage
api_key = ""  # Your actual API key here

# Initialize the language agent
llm = OpenAIChat(
    openai_api_key=api_key,
    temperature=0.5,
    max_tokens=3000,
)

# Initialize Agents for individual tasks
flow1 = Agent(llm=llm, max_loops=1, dashboard=False)
flow2 = Agent(llm=llm, max_loops=1, dashboard=False)

# Create the Sequential Workflow
workflow = SequentialWorkflow(max_loops=1)

# Add tasks to the workflow
workflow.add("Generate a 10,000 word blog on health and wellness.", flow1)
workflow.add("Summarize the generated blog", flow2)

# Run the workflow
workflow.run()

# Get and display the results of each task in the workflow
results = workflow.get_task_results()
for task_description, result in results.items():
    print(f"Task: {task_description}, Result: {result}")
```

In this example, we create a Sequential Workflow, add two tasks to it, run the workflow, and then retrieve and display the results of each task.

### Example 4: Removing a Task from a Sequential Workflow

In this example, we'll create a Sequential Workflow, add tasks to it, and then remove a specific task from the workflow.

```python
from swarms.models import OpenAIChat
from swarms.structs import Agent
from swarms.structs.sequential_workflow import SequentialWorkflow

# Example usage
api_key = ""  # Your actual API key here

# Initialize the language agent
llm = OpenAIChat(
    openai_api_key=api_key,
    temperature=0.5,
    max_tokens=3000,
)

# Initialize Agents for individual tasks
flow1 = Agent(llm=llm, max_loops=1, dashboard=False)
flow2 = Agent(llm=llm, max_loops=1, dashboard=False)

# Create the Sequential Workflow
workflow = SequentialWorkflow(max_loops=1)

# Add tasks to the workflow
workflow.add("Generate a 10,000 word blog on health and wellness.", flow1)
workflow.add("Summarize the generated blog", flow2)

# Remove a specific task from the workflow
workflow.remove_task("Generate a 10,000 word blog on health and wellness.")

# Output the list of tasks in the workflow after removal
print("Tasks in the workflow after removing a task:")
for task in workflow.tasks:
    print(f"Task: {task.description}")
```

In this example, we create a Sequential Workflow, add two tasks to it, and then remove a specific task from the workflow.

### Example 5: Updating Task Arguments in a Sequential Workflow

In this example, we'll create a Sequential Workflow, add tasks to it, and then update the arguments of a specific task in the workflow.

```python
from swarms.models import OpenAIChat
from swarms.structs import Agent
from swarms.structs.sequential_workflow import SequentialWorkflow

# Example usage
api_key = (
    ""  # Your actual API key here
)

# Initialize the language agent
llm = OpenAIChat(
    openai_api_key=api_key,
    temperature=0.5,
    max_tokens=3000,
)

# Initialize Agents for individual tasks
flow1 = Agent(llm=llm, max_loops=1, dashboard=False)
flow2 = Agent(llm=llm, max_loops=1, dashboard=False)

# Create the Sequential Workflow
workflow = SequentialWorkflow(max_loops=1)

# Add tasks to the workflow
workflow.add("Generate a 10,000 word blog on health and wellness.", flow1)
workflow.add("Summarize the generated blog", flow2)

# Update the arguments of a specific task in the workflow
workflow.update_task("Generate a 10,000 word blog on health and wellness.", max_loops=2)

# Output the list of tasks in the workflow after updating task arguments
print("Tasks in the workflow after updating task arguments:")
for task in workflow.tasks:
    print(f"Task: {task.description}, Arguments: {

task.arguments}")
```

In this example, we create a Sequential Workflow, add two tasks to it, and then update the arguments of a specific task in the workflow.

These examples demonstrate various operations and use cases for working with a Sequential Workflow.

# Why `SequentialWorkflow`?

## Enhancing Autonomous Agent Development

The development of autonomous agents, whether they are conversational AI, robotic systems, or any other AI-driven application, often involves complex workflows that require a sequence of tasks to be executed in a specific order. Managing and orchestrating these tasks efficiently is crucial for building reliable and effective agents. The Sequential Workflow module serves as a valuable tool for AI engineers in achieving this goal.

## Reliability and Coordination

One of the primary challenges in autonomous agent development is ensuring that tasks are executed in the correct sequence and that the results of one task can be used as inputs for subsequent tasks. The Sequential Workflow module simplifies this process by allowing AI engineers to define and manage workflows in a structured and organized manner.

By using the Sequential Workflow module, AI engineers can achieve the following benefits:

### 1. Improved Reliability

Reliability is a critical aspect of autonomous agents. The ability to handle errors gracefully and recover from failures is essential for building robust systems. The Sequential Workflow module offers a systematic approach to task execution, making it easier to handle errors, retry failed tasks, and ensure that the agent continues to operate smoothly.

### 2. Task Coordination

Coordinating tasks in the correct order is essential for achieving the desired outcome. The Sequential Workflow module enforces task sequencing, ensuring that each task is executed only when its dependencies are satisfied. This eliminates the risk of executing tasks out of order, which can lead to incorrect results.

### 3. Code Organization

Managing complex workflows can become challenging without proper organization. The Sequential Workflow module encourages AI engineers to structure their code in a modular and maintainable way. Each task can be encapsulated as a separate unit, making it easier to understand, modify, and extend the agent's behavior.

### 4. Workflow Visualization

Visualization is a powerful tool for understanding and debugging workflows. The Sequential Workflow module can be extended to include a visualization dashboard, allowing AI engineers to monitor the progress of tasks, track results, and identify bottlenecks or performance issues.

## TODO: Future Features

While the Sequential Workflow module offers significant advantages, there are opportunities for further enhancement. Here is a list of potential features and improvements that can be added to make it even more versatile and adaptable for various AI engineering tasks:

### 1. Asynchronous Support

Adding support for asynchronous task execution can improve the efficiency of workflows, especially when dealing with tasks that involve waiting for external events or resources.

### 2. Context Managers

Introducing context manager support for tasks can simplify resource management, such as opening and closing files, database connections, or network connections within a task's context.

### 3. Workflow History

Maintaining a detailed history of workflow execution, including timestamps, task durations, and input/output data, can facilitate debugging and performance analysis.

### 4. Parallel Processing

Enhancing the module to support parallel processing with a pool of workers can significantly speed up the execution of tasks, especially for computationally intensive workflows.

### 5. Error Handling Strategies

Providing built-in error handling strategies, such as retries, fallbacks, and custom error handling functions, can make the module more robust in handling unexpected failures.

## Conclusion

The Sequential Workflow module is a valuable tool for AI engineers working on autonomous agents and complex AI-driven applications. It offers a structured and reliable approach to defining and executing workflows, ensuring that tasks are performed in the correct sequence. By using this module, AI engineers can enhance the reliability, coordination, and maintainability of their agents.

As the field of AI continues to evolve, the demand for efficient workflow management tools will only increase. The Sequential Workflow module is a step towards meeting these demands and empowering AI engineers to create more reliable and capable autonomous agents. With future enhancements and features, it has the potential to become an indispensable asset in the AI engineer's toolkit.

In summary, the Sequential Workflow module provides a foundation for orchestrating complex tasks and workflows, enabling AI engineers to focus on designing intelligent agents that can perform tasks with precision and reliability.


## Frequently Asked Questions (FAQs)

### Q1: What is the difference between a task and a agent in Sequential Workflows?

**A1:** In Sequential Workflows, a **task** refers to a specific unit of work that needs to be executed. It can be implemented as a callable object, such as a Python function, and is the fundamental building block of a workflow.

A **agent**, on the other hand, is an encapsulation of a task within the workflow. Agents define the order in which tasks are executed and can be thought of as task containers. They allow you to specify dependencies, error handling, and other workflow-related configurations.

### Q2: Can I run tasks in parallel within a Sequential Workflow?

**A2:** Yes, you can run tasks in parallel within a Sequential Workflow by using parallel execution techniques. This advanced feature allows you to execute multiple tasks concurrently, improving performance and efficiency. You can explore this feature further in the guide's section on "Parallel Execution."

### Q3: How do I handle errors within Sequential Workflows?

**A3:** Error handling within Sequential Workflows can be implemented by adding error-handling logic within your task functions. You can catch exceptions and handle errors gracefully, ensuring that your workflow can recover from unexpected scenarios. The guide also covers more advanced error handling strategies, such as retrying failed tasks and handling specific error types.

### Q4: What are some real-world use cases for Sequential Workflows?

**A4:** Sequential Workflows can be applied to a wide range of real-world use cases, including:

- **Data ETL (Extract, Transform, Load) Processes:** Automating data pipelines that involve data extraction, transformation, and loading into databases or data warehouses.

- **Batch Processing:** Running batch jobs that process large volumes of data or perform data analysis.

- **Automation of DevOps Tasks:** Streamlining DevOps processes such as deployment, provisioning, and monitoring.

- **Cross-system Integrations:** Automating interactions between different systems, services, or APIs.

- **Report Generation:** Generating reports and documents automatically based on data inputs.

- **Workflow Orchestration:** Orchestrating complex workflows involving multiple steps and dependencies.

- **Resource Provisioning:** Automatically provisioning and managing cloud resources.

These are just a few examples, and Sequential Workflows can be tailored to various automation needs across industries.
