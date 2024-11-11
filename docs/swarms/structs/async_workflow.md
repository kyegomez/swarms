# AsyncWorkflow Documentation

The `AsyncWorkflow` class represents an asynchronous workflow designed to execute tasks concurrently. This class is ideal for scenarios where tasks need to be run asynchronously, leveraging Python's asyncio capabilities to manage multiple tasks efficiently.

### Key Concepts

- **Asynchronous Execution**: Tasks are run concurrently using asyncio, allowing for non-blocking operations.
- **Task Pool**: A collection of tasks to be executed within the workflow.
- **Event Loop**: The asyncio event loop that manages the execution of asynchronous tasks.
- **Stopping Condition**: A condition that, when met, stops the execution of the workflow.

## Attributes

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `name` | `str` | `"Async Workflow"` | The name of the workflow. |
| `description` | `str` | `"A workflow to run asynchronous tasks"` | The description of the workflow. |
| `max_loops` | `int` | `1` | The maximum number of loops to run the workflow. |
| `autosave` | `bool` | `True` | Flag indicating whether to autosave the results. |
| `dashboard` | `bool` | `False` | Flag indicating whether to display a dashboard. |
| `task_pool` | `List[Any]` | `[]` | The list of tasks in the workflow. |
| `results` | `List[Any]` | `[]` | The list of results from running the tasks. |
| `loop` | `Optional[asyncio.AbstractEventLoop]` | `None` | The event loop to use. |
| `stopping_condition` | `Optional[Callable]` | `None` | The stopping condition for the workflow. |
| `agents` | `List[Agent]` | `None` | A list of agents participating in the workflow. |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | The name of the workflow. |
| `description` | `str` | The description of the workflow. |
| `max_loops` | `int` | The maximum number of loops to run the workflow. |
| `autosave` | `bool` | Flag indicating whether to autosave the results. |
| `dashboard` | `bool` | Flag indicating whether to display a dashboard. |
| `task_pool` | `List[Any]` | The list of tasks in the workflow. |
| `results` | `List[Any]` | The list of results from running the tasks. |
| `loop` | `Optional[asyncio.AbstractEventLoop]` | The event loop to use. |
| `stopping_condition` | `Optional[Callable]` | The stopping condition for the workflow. |
| `agents` | `List[Agent]` | A list of agents participating in the workflow. |

## Methods

### add

Adds a task or a list of tasks to the task pool.

**Arguments:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | `Any` | `None` | A single task to add. |
| `tasks` | `List[Any]` | `None` | A list of tasks to add. |

**Raises:**

- `ValueError`: If neither task nor tasks are provided.

**Examples:**

```python
workflow = AsyncWorkflow()
task1 = Task(description="Task 1")
task2 = Task(description="Task 2")

# Adding a single task
await workflow.add(task=task1)

# Adding multiple tasks
await workflow.add(tasks=[task1, task2])
```

### delete

Deletes a task from the workflow.

**Arguments:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | `Any` | `None` | A single task to delete. |
| `tasks` | `List[Task]` | `None` | A list of tasks to delete. |

**Examples:**

```python
workflow = AsyncWorkflow()
task1 = Task(description="Task 1")
task2 = Task(description="Task 2")

# Adding tasks to the workflow
await workflow.add(tasks=[task1, task2])

# Deleting a single task
await workflow.delete(task=task1)

# Deleting multiple tasks
await workflow.delete(tasks=[task1, task2])
```

### run

Runs the workflow and returns the results.

**Returns:**

| Return Type | Description |
|-------------|-------------|
| `List[Any]` | The results of the executed tasks. |

**Examples:**

```python
workflow = AsyncWorkflow()
task1 = Task(description="Task 1", execute=async_function)
task2 = Task(description="Task 2", execute=async_function)

# Adding tasks to the workflow
await workflow.add(tasks=[task1, task2])

# Running the workflow
results = await workflow.run()
```

### Additional Examples

#### Example 1: Simple AsyncWorkflow

```python
import asyncio
from swarms.structs.agent import Agent
from swarms.structs.task import Task

async def simple_task():
    await asyncio.sleep(1)
    return "Task Completed"

workflow = AsyncWorkflow()
task = Task(description="Simple Task", execute=simple_task)

# Adding a task to the workflow
await workflow.add(task=task)

# Running the workflow
results = await workflow.run()
print(results)  # Output: ["Task Completed"]
```

#### Example 2: Workflow with Multiple Tasks

```python
import asyncio
from swarms.structs.agent import Agent
from swarms.structs.task import Task

async def task1():
    await asyncio.sleep(1)
    return "Task 1 Completed"

async def task2():
    await asyncio.sleep(2)
    return "Task 2 Completed"

workflow = AsyncWorkflow()
task_1 = Task(description="Task 1", execute=task1)
task_2 = Task(description="Task 2", execute=task2)

# Adding tasks to the workflow
await workflow.add(tasks=[task_1, task_2])

# Running the workflow
results = await workflow.run()
print(results)  # Output: ["Task 1 Completed", "Task 2 Completed"]
```

#### Example 3: Workflow with Stopping Condition

```python
import asyncio
from swarms.structs.agent import Agent
from swarms.structs.task import Task

async def task1():
    await asyncio.sleep(1)
    return "Task 1 Completed"

async def task2():
    await asyncio.sleep(2)
    return "Task 2 Completed"

def stop_condition(results):
    return "Task 2 Completed" in results

workflow = AsyncWorkflow(stopping_condition=stop_condition)
task_1 = Task(description="Task 1", execute=task1)
task_2 = Task(description="Task 2", execute=task2)

# Adding tasks to the workflow
await workflow.add(tasks=[task_1, task_2])

# Running the workflow
results = await workflow.run()
print(results)  # Output: ["Task 1 Completed", "Task 2 Completed"]
```

# Async Workflow

The AsyncWorkflow allows multiple agents to process tasks concurrently using Python's asyncio framework.

## Usage Example

```python
import asyncio
from swarms import Agent, AsyncWorkflow
from swarm_models import OpenAIChat

# Initialize model
model = OpenAIChat(
    openai_api_key="your-api-key",
    model_name="gpt-4",
    temperature=0.7
)

# Create agents
agents = [
    Agent(
        agent_name=f"Analysis-Agent-{i}",
        llm=model,
        max_loops=1,
        dashboard=False,
        verbose=True,
    )
    for i in range(3)
]

# Initialize workflow
workflow = AsyncWorkflow(
    name="Analysis-Workflow",
    agents=agents,
    max_workers=3,
    verbose=True
)

# Run workflow
async def main():
    task = "Analyze the potential impact of AI on healthcare"
    results = await workflow.run(task)
    for i, result in enumerate(results):
        print(f"Agent {i} result: {result}")

# Execute
asyncio.run(main())
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | "AsyncWorkflow" | Name of the workflow |
| `agents` | List[Agent] | None | List of agents to execute tasks |
| `max_workers` | int | 5 | Maximum number of concurrent workers |
| `dashboard` | bool | False | Enable/disable dashboard |
| `autosave` | bool | False | Enable/disable autosaving results |
| `verbose` | bool | False | Enable/disable verbose logging |