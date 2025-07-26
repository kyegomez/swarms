# ConcurrentWorkflow Documentation

## Overview

The `ConcurrentWorkflow` class is designed to facilitate the concurrent execution of multiple agents, each tasked with solving a specific query or problem. This class is particularly useful in scenarios where multiple agents need to work in parallel, allowing for efficient resource utilization and faster completion of tasks. The workflow manages the execution, collects metadata, and optionally saves the results in a structured format.

### Key Features

- **Concurrent Execution**: Runs multiple agents simultaneously using Python's `ThreadPoolExecutor`
- **Interactive Mode**: Supports interactive task modification and execution
- **Caching System**: Implements LRU caching for repeated prompts
- **Progress Tracking**: Optional progress bar for task execution
- **Enhanced Error Handling**: Implements retry mechanism with exponential backoff
- **Input Validation**: Validates task inputs before execution
- **Batch Processing**: Supports running tasks in batches
- **Metadata Collection**: Gathers detailed metadata about each agent's execution
- **Customizable Output**: Allows saving metadata to file or returning as string/dictionary

## Class Definition

### Attributes

| Attribute              | Type                    | Description                                               |
|------------------------|-------------------------|-----------------------------------------------------------|
| `name`                 | `str`                   | The name of the workflow. Defaults to `"ConcurrentWorkflow"`. |
| `description`          | `str`                   | A brief description of the workflow.                      |
| `agents`               | `List[Agent]`           | A list of agents to be executed concurrently.             |
| `metadata_output_path` | `str`                   | Path to save the metadata output. Defaults to `"agent_metadata.json"`. |
| `auto_save`            | `bool`                  | Flag indicating whether to automatically save the metadata. |
| `output_type`          | `str`                   | The type of output format. Defaults to `"dict"`.          |
| `max_loops`            | `int`                   | Maximum number of loops for each agent. Defaults to `1`.  |
| `return_str_on`        | `bool`                  | Flag to return output as string. Defaults to `False`.     |
| `auto_generate_prompts`| `bool`                  | Flag indicating whether to auto-generate prompts for agents. |
| `return_entire_history`| `bool`                  | Flag to return entire conversation history. Defaults to `False`. |
| `interactive`          | `bool`                  | Flag indicating whether to enable interactive mode. Defaults to `False`. |
| `cache_size`           | `int`                   | The size of the cache. Defaults to `100`.                 |
| `max_retries`          | `int`                   | The maximum number of retry attempts. Defaults to `3`.    |
| `retry_delay`          | `float`                 | The delay between retry attempts in seconds. Defaults to `1.0`. |
| `show_progress`        | `bool`                  | Flag indicating whether to show progress. Defaults to `False`. |
| `_cache`               | `dict`                  | The cache for storing agent outputs.                      |
| `_progress_bar`        | `tqdm`                  | The progress bar for tracking execution.                  |

## Methods

### ConcurrentWorkflow.\_\_init\_\_

Initializes the `ConcurrentWorkflow` class with the provided parameters.

#### Parameters

| Parameter             | Type           | Default Value                          | Description                                               |
|-----------------------|----------------|----------------------------------------|-----------------------------------------------------------|
| `name`                | `str`          | `"ConcurrentWorkflow"`                 | The name of the workflow.                                 |
| `description`         | `str`          | `"Execution of multiple agents concurrently"` | A brief description of the workflow.               |
| `agents`              | `List[Agent]`  | `[]`                                   | A list of agents to be executed concurrently.             |
| `metadata_output_path`| `str`          | `"agent_metadata.json"`                | Path to save the metadata output.                         |
| `auto_save`           | `bool`         | `True`                                 | Flag indicating whether to automatically save the metadata. |
| `output_type`         | `str`          | `"dict"`                               | The type of output format.                                |
| `max_loops`           | `int`          | `1`                                    | Maximum number of loops for each agent.                   |
| `return_str_on`       | `bool`         | `False`                                | Flag to return output as string.                          |
| `auto_generate_prompts`| `bool`        | `False`                                | Flag indicating whether to auto-generate prompts for agents. |
| `return_entire_history`| `bool`        | `False`                                | Flag to return entire conversation history.               |
| `interactive`         | `bool`         | `False`                                | Flag indicating whether to enable interactive mode.        |
| `cache_size`          | `int`          | `100`                                  | The size of the cache.                                    |
| `max_retries`         | `int`          | `3`                                    | The maximum number of retry attempts.                     |
| `retry_delay`         | `float`        | `1.0`                                  | The delay between retry attempts in seconds.              |
| `show_progress`       | `bool`         | `False`                                | Flag indicating whether to show progress.                 |

#### Raises

- `ValueError`: If the list of agents is empty or if the description is empty.

### ConcurrentWorkflow.disable_agent_prints

Disables print statements for all agents in the workflow.

```python
workflow.disable_agent_prints()
```

### ConcurrentWorkflow.activate_auto_prompt_engineering

Activates the auto-generate prompts feature for all agents in the workflow.

```python
workflow.activate_auto_prompt_engineering()
```

### ConcurrentWorkflow.enable_progress_bar

Enables the progress bar display for task execution.

```python
workflow.enable_progress_bar()
```

### ConcurrentWorkflow.disable_progress_bar

Disables the progress bar display.

```python
workflow.disable_progress_bar()
```

### ConcurrentWorkflow.clear_cache

Clears the task cache.

```python
workflow.clear_cache()
```

### ConcurrentWorkflow.get_cache_stats

Gets cache statistics.

#### Returns

- `Dict[str, int]`: A dictionary containing cache statistics.

```python
stats = workflow.get_cache_stats()
print(stats)  # {'cache_size': 5, 'max_cache_size': 100}
```

### ConcurrentWorkflow.run

Executes the workflow for the provided task.

#### Parameters

| Parameter   | Type                | Description                                               |
|-------------|---------------------|-----------------------------------------------------------|
| `task`      | `Optional[str]`     | The task or query to give to all agents.                  |
| `img`       | `Optional[str]`     | The image to be processed by the agents.                  |
| `*args`     | `tuple`             | Additional positional arguments.                          |
| `**kwargs`  | `dict`              | Additional keyword arguments.                             |

#### Returns

- `Any`: The result of the execution, format depends on output_type and return_entire_history settings.

#### Raises

- `ValueError`: If an invalid device is specified.
- `Exception`: If any other error occurs during execution.

### ConcurrentWorkflow.run_batched

Runs the workflow for a batch of tasks.

#### Parameters

| Parameter   | Type         | Description                                               |
|-------------|--------------|-----------------------------------------------------------|
| `tasks`     | `List[str]`  | A list of tasks or queries to give to all agents.         |

#### Returns

- `List[Any]`: A list of results for each task.

## Usage Examples

### Example 1: Basic Usage with Interactive Mode

```python
from swarms import Agent, ConcurrentWorkflow

# Initialize agents
agents = [
    Agent(
        agent_name=f"Agent-{i}",
        system_prompt="You are a helpful assistant.",
        model_name="gpt-4",
        max_loops=1,
    )
    for i in range(3)
]

# Initialize workflow with interactive mode
workflow = ConcurrentWorkflow(
    name="Interactive Workflow",
    agents=agents,
    interactive=True,
    show_progress=True,
    cache_size=100,
    max_retries=3,
    retry_delay=1.0
)

# Run workflow
task = "What are the benefits of using Python for data analysis?"
result = workflow.run(task)
print(result)
```

### Example 2: Batch Processing with Progress Bar

```python
# Initialize workflow
workflow = ConcurrentWorkflow(
    name="Batch Processing Workflow",
    agents=agents,
    show_progress=True,
    auto_save=True
)

# Define tasks
tasks = [
    "Analyze the impact of climate change on agriculture",
    "Evaluate renewable energy solutions",
    "Assess water conservation strategies"
]

# Run batch processing
results = workflow.run_batched(tasks)

# Process results
for task, result in zip(tasks, results):
    print(f"Task: {task}")
    print(f"Result: {result}\n")
```

### Example 3: Error Handling and Retries

```python
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize workflow with retry settings
workflow = ConcurrentWorkflow(
    name="Reliable Workflow",
    agents=agents,
    max_retries=3,
    retry_delay=1.0,
    show_progress=True
)

# Run workflow with error handling
try:
    task = "Generate a comprehensive market analysis report"
    result = workflow.run(task)
    print(result)
except Exception as e:
    logging.error(f"An error occurred: {str(e)}")
```

## Tips and Best Practices

- **Agent Initialization**: Ensure all agents are correctly initialized with required configurations.
- **Interactive Mode**: Use interactive mode for tasks requiring user input or modification.
- **Caching**: Utilize the caching system for repeated tasks to improve performance.
- **Progress Tracking**: Enable progress bar for long-running tasks to monitor execution.
- **Error Handling**: Implement proper error handling and use retry mechanism for reliability.
- **Resource Management**: Monitor cache size and clear when necessary.
- **Batch Processing**: Use batch processing for multiple related tasks.
- **Logging**: Implement detailed logging for debugging and monitoring.

## References and Resources

- [Python's ThreadPoolExecutor Documentation](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor)
- [tqdm Progress Bar Documentation](https://tqdm.github.io/)
- [Python's functools.lru_cache Documentation](https://docs.python.org/3/library/functools.html#functools.lru_cache)
- [Loguru for Logging in Python](https://loguru.readthedocs.io/en/stable/)