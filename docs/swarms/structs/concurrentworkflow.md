# ConcurrentWorkflow Documentation

The `ConcurrentWorkflow` class is designed to facilitate the concurrent execution of multiple agents, each tasked with solving a specific query or problem. This class is particularly useful in scenarios where multiple agents need to work in parallel, allowing for efficient resource utilization and faster completion of tasks. The workflow manages the execution, handles streaming callbacks, and provides optional dashboard monitoring for real-time progress tracking.

Full Path: `swarms.structs.concurrent_workflow`

### Key Features

| Feature                   | Description                                                                                   |
|---------------------------|-----------------------------------------------------------------------------------------------|
| Concurrent Execution      | Runs multiple agents simultaneously using Python's `ThreadPoolExecutor`                       |
| Dashboard Monitoring      | Optional real-time dashboard for tracking agent status and progress                           |
| Streaming Support         | Full support for streaming callbacks during agent execution                                   |
| Error Handling            | Comprehensive error handling with logging and status tracking                                 |
| Batch Processing          | Supports running multiple tasks sequentially                                                  |
| Resource Management       | Automatic cleanup of resources and connections                                                |
| Flexible Output Types     | Multiple output format options for conversation history                                       |
| Agent Status Tracking     | Real-time tracking of agent execution states (pending, running, completed, error)             |

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

## Constructor

### ConcurrentWorkflow.\_\_init\_\_

Initializes the `ConcurrentWorkflow` class with the provided parameters.

#### Parameters

| Parameter             | Type                          | Default Value                          | Description                                               |
|-----------------------|-------------------------------|----------------------------------------|-----------------------------------------------------------|
| `id`                  | `str`                         | `swarm_id()`                           | Unique identifier for the workflow instance.              |
| `name`                | `str`                         | `"ConcurrentWorkflow"`                 | The name of the workflow.                                 |
| `description`         | `str`                         | `"Execution of multiple agents concurrently"` | A brief description of the workflow.               |
| `agents`              | `List[Union[Agent, Callable]]`| `None`                                 | A list of agents or callables to be executed concurrently. |
| `auto_save`           | `bool`                        | `True`                                 | Flag indicating whether to automatically save metadata.  |
| `output_type`         | `str`                         | `"dict-all-except-first"`              | The type of output format.                                |
| `max_loops`           | `int`                         | `1`                                    | Maximum number of loops for each agent.                   |
| `auto_generate_prompts`| `bool`                       | `False`                                | Flag indicating whether to auto-generate prompts for agents. |
| `show_dashboard`      | `bool`                        | `False`                                | Flag indicating whether to show real-time dashboard.     |

#### Raises

- `ValueError`: If no agents are provided or if the agents list is empty.

## Methods

### ConcurrentWorkflow.fix_agents

Configures agents for dashboard mode by disabling print statements when dashboard is enabled.

#### Returns

- `List[Union[Agent, Callable]]`: The configured list of agents.

```python
agents = workflow.fix_agents()
```

### ConcurrentWorkflow.reliability_check

Validates workflow configuration and ensures agents are properly set up.

#### Raises

- `ValueError`: If no agents are provided or if the agents list is empty.

```python
workflow.reliability_check()
```

### ConcurrentWorkflow.activate_auto_prompt_engineering

Enables automatic prompt generation for all agents in the workflow.

```python
workflow.activate_auto_prompt_engineering()
```

### ConcurrentWorkflow.display_agent_dashboard

Displays real-time dashboard showing agent status and progress.

#### Parameters

| Parameter   | Type    | Default Value              | Description                                               |
|-------------|---------|----------------------------|-----------------------------------------------------------|
| `title`     | `str`   | `"ConcurrentWorkflow Dashboard"` | Title for the dashboard.                          |
| `is_final`  | `bool`  | `False`                    | Whether this is the final dashboard display.             |

```python
workflow.display_agent_dashboard("Execution Progress", is_final=False)
```

### ConcurrentWorkflow.run_with_dashboard

Executes agents with real-time dashboard monitoring and streaming support.

#### Parameters

| Parameter             | Type                              | Description                                               |
|-----------------------|-----------------------------------|-----------------------------------------------------------|
| `task`                | `str`                             | The task to execute.                                       |
| `img`                 | `Optional[str]`                   | Optional image for processing.                            |
| `imgs`                | `Optional[List[str]]`             | Optional list of images for processing.                   |
| `streaming_callback`  | `Optional[Callable[[str, str, bool], None]]` | Callback for streaming agent outputs.         |

#### Returns

- `Any`: The formatted conversation history based on output_type.

```python
result = workflow.run_with_dashboard(
    task="Analyze this data",
    streaming_callback=lambda agent, chunk, done: print(f"{agent}: {chunk}")
)
```

### ConcurrentWorkflow._run

Executes agents concurrently without dashboard monitoring.

#### Parameters

| Parameter             | Type                              | Description                                               |
|-----------------------|-----------------------------------|-----------------------------------------------------------|
| `task`                | `str`                             | The task to execute.                                       |
| `img`                 | `Optional[str]`                   | Optional image for processing.                            |
| `imgs`                | `Optional[List[str]]`             | Optional list of images for processing.                   |
| `streaming_callback`  | `Optional[Callable[[str, str, bool], None]]` | Callback for streaming agent outputs.         |

#### Returns

- `Any`: The formatted conversation history based on output_type.

```python
result = workflow._run(
    task="Process this task",
    streaming_callback=lambda agent, chunk, done: print(f"{agent}: {chunk}")
)
```

### ConcurrentWorkflow._run_agent_with_streaming

Runs a single agent with streaming callback support.

#### Parameters

| Parameter             | Type                              | Description                                               |
|-----------------------|-----------------------------------|-----------------------------------------------------------|
| `agent`               | `Union[Agent, Callable]`          | The agent or callable to execute.                         |
| `task`                | `str`                             | The task to execute.                                       |
| `img`                 | `Optional[str]`                   | Optional image for processing.                            |
| `imgs`                | `Optional[List[str]]`             | Optional list of images for processing.                   |
| `streaming_callback`  | `Optional[Callable[[str, str, bool], None]]` | Callback for streaming outputs.                |

#### Returns

- `str`: The output from the agent execution.

```python
output = workflow._run_agent_with_streaming(
    agent=my_agent,
    task="Analyze data",
    streaming_callback=lambda agent, chunk, done: print(f"{agent}: {chunk}")
)
```

### ConcurrentWorkflow.cleanup

Cleans up resources and connections used by the workflow.

```python
workflow.cleanup()
```

### ConcurrentWorkflow.run

Main execution method that runs all agents concurrently.

#### Parameters

| Parameter             | Type                              | Description                                               |
|-----------------------|-----------------------------------|-----------------------------------------------------------|
| `task`                | `str`                             | The task to execute.                                       |
| `img`                 | `Optional[str]`                   | Optional image for processing.                            |
| `imgs`                | `Optional[List[str]]`             | Optional list of images for processing.                   |
| `streaming_callback`  | `Optional[Callable[[str, str, bool], None]]` | Callback for streaming agent outputs.         |

#### Returns

- `Any`: The formatted conversation history based on output_type.

```python
result = workflow.run(
    task="What are the benefits of renewable energy?",
    streaming_callback=lambda agent, chunk, done: print(f"{agent}: {chunk}")
)
```

### ConcurrentWorkflow.batch_run

Executes the workflow on multiple tasks sequentially.

#### Parameters

| Parameter             | Type                              | Description                                               |
|-----------------------|-----------------------------------|-----------------------------------------------------------|
| `tasks`               | `List[str]`                       | List of tasks to execute.                                  |
| `imgs`                | `Optional[List[str]]`             | Optional list of images corresponding to tasks.           |
| `streaming_callback`  | `Optional[Callable[[str, str, bool], None]]` | Callback for streaming outputs.                |

#### Returns

- `List[Any]`: List of results for each task.

```python
results = workflow.batch_run(
    tasks=["Task 1", "Task 2", "Task 3"],
    streaming_callback=lambda agent, chunk, done: print(f"{agent}: {chunk}")
)
```

## Usage Examples

### Example 1: Basic Concurrent Execution

```python
from swarms import Agent, ConcurrentWorkflow

# Initialize agents
agents = [
    Agent(
        agent_name="Research-Agent",
        system_prompt="You are a research specialist focused on gathering information.",
        model_name="gpt-4",
        max_loops=1,
    ),
    Agent(
        agent_name="Analysis-Agent",
        system_prompt="You are an analysis expert who synthesizes information.",
        model_name="gpt-4",
        max_loops=1,
    ),
    Agent(
        agent_name="Summary-Agent",
        system_prompt="You are a summarization expert who creates concise reports.",
        model_name="gpt-4",
        max_loops=1,
    )
]

# Initialize workflow
workflow = ConcurrentWorkflow(
    name="Research Analysis Workflow",
    description="Concurrent execution of research, analysis, and summarization tasks",
    agents=agents,
    auto_save=True,
    output_type="dict-all-except-first",
    show_dashboard=False
)

# Run workflow
task = "What are the environmental impacts of electric vehicles?"
result = workflow.run(task)
print(result)
```

### Example 2: Dashboard Monitoring with Streaming

```python
import time

def streaming_callback(agent_name: str, chunk: str, is_final: bool):
    """Handle streaming output from agents."""
    if chunk:
        print(f"[{agent_name}] {chunk}", end="", flush=True)
    if is_final:
        print(f"\n[{agent_name}] Completed\n")

# Initialize workflow with dashboard
workflow = ConcurrentWorkflow(
    name="Monitored Workflow",
    agents=agents,
    show_dashboard=True,  # Enable real-time dashboard
    output_type="dict-all-except-first"
)

# Run with streaming and dashboard
task = "Analyze the future of artificial intelligence in healthcare"
result = workflow.run(
    task=task,
    streaming_callback=streaming_callback
)

print("Final Result:", result)
```

### Example 3: Batch Processing Multiple Tasks

```python
# Define multiple tasks
tasks = [
    "What are the benefits of renewable energy adoption?",
    "How does blockchain technology impact supply chains?",
    "What are the challenges of implementing remote work policies?",
    "Analyze the growth of e-commerce in developing countries"
]

# Initialize workflow for batch processing
workflow = ConcurrentWorkflow(
    name="Batch Analysis Workflow",
    agents=agents,
    output_type="dict-all-except-first",
    show_dashboard=False
)

# Process all tasks
results = workflow.batch_run(tasks=tasks)

# Display results
for i, (task, result) in enumerate(zip(tasks, results)):
    print(f"\n{'='*50}")
    print(f"Task {i+1}: {task}")
    print(f"{'='*50}")
    print(f"Result: {result}")
```

### Example 4: Auto-Prompt Engineering

```python
# Initialize agents without specific prompts
agents = [
    Agent(
        agent_name="Creative-Agent",
        model_name="gpt-4",
        max_loops=1,
    ),
    Agent(
        agent_name="Technical-Agent",
        model_name="gpt-4",
        max_loops=1,
    )
]

# Initialize workflow with auto-prompt engineering
workflow = ConcurrentWorkflow(
    name="Auto-Prompt Workflow",
    agents=agents,
    auto_generate_prompts=True,  # Enable auto-prompt generation
    output_type="dict-all-except-first"
)

# Activate auto-prompt engineering (can also be done in init)
workflow.activate_auto_prompt_engineering()

# Run workflow
task = "Design a mobile app for fitness tracking"
result = workflow.run(task)
print(result)
```

### Example 5: Error Handling and Cleanup

```python
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize workflow
workflow = ConcurrentWorkflow(
    name="Reliable Workflow",
    agents=agents,
    output_type="dict-all-except-first"
)

# Run workflow with proper error handling
try:
    task = "Generate a comprehensive report on quantum computing applications"
    result = workflow.run(task)
    print("Workflow completed successfully!")
    print(result)
except Exception as e:
    logging.error(f"Workflow failed: {str(e)}")
finally:
    # Always cleanup resources
    workflow.cleanup()
    print("Resources cleaned up")
```

### Example 6: Working with Images

```python
# Initialize agents capable of image processing
vision_agents = [
    Agent(
        agent_name="Image-Analysis-Agent",
        system_prompt="You are an expert at analyzing images and extracting insights.",
        model_name="gpt-4-vision-preview",
        max_loops=1,
    ),
    Agent(
        agent_name="Content-Description-Agent",
        system_prompt="You specialize in creating detailed descriptions of visual content.",
        model_name="gpt-4-vision-preview",
        max_loops=1,
    )
]

# Initialize workflow for image processing
workflow = ConcurrentWorkflow(
    name="Image Analysis Workflow",
    agents=vision_agents,
    output_type="dict-all-except-first",
    show_dashboard=True
)

# Run with image input
task = "Analyze this image and provide insights about its content"
image_path = "/path/to/image.jpg"

result = workflow.run(
    task=task,
    img=image_path,
    streaming_callback=lambda agent, chunk, done: print(f"{agent}: {chunk}")
)

print(result)
```

### Example 7: Custom Callable Agents

```python
from typing import Optional

def custom_analysis_agent(task: str, img: Optional[str] = None, **kwargs) -> str:
    """Custom analysis function that can be used as an agent."""
    # Custom logic here
    return f"Custom analysis result for: {task}"

def sentiment_analysis_agent(task: str, img: Optional[str] = None, **kwargs) -> str:
    """Sentiment analysis function."""
    # Custom sentiment analysis logic
    return f"Sentiment analysis for: {task}"

# Mix of Agent objects and callable functions
mixed_agents = [
    Agent(
        agent_name="GPT-Agent",
        system_prompt="You are a helpful assistant.",
        model_name="gpt-4",
        max_loops=1,
    ),
    custom_analysis_agent,  # Callable function
    sentiment_analysis_agent  # Another callable function
]

# Initialize workflow with mixed agent types
workflow = ConcurrentWorkflow(
    name="Mixed Agents Workflow",
    agents=mixed_agents,
    output_type="dict-all-except-first"
)

# Run workflow
task = "Analyze customer feedback and provide insights"
result = workflow.run(task)
print(result)
```